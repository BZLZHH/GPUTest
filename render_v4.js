// --- 全局变量 ---
const canvas = document.getElementById('canvas');
const backendSelector = document.getElementById('backend-selector');
const shaderSelector = document.getElementById('shader-selector');
const bouncesSlider = document.getElementById('bounces-slider');
const shadowSlider = document.getElementById('shadow-slider');
const stepsSlider = document.getElementById('steps-slider');
const resolutionSlider = document.getElementById('resolution-slider');
const shaderPresets = document.querySelectorAll('.shader-btn');
const previewItems = document.querySelectorAll('.preview-item');
const loadingScreen = document.getElementById('loading');

const bouncesValueSpan = document.getElementById('bounces-value');
const shadowValueSpan = document.getElementById('shadow-value');
const stepsValueSpan = document.getElementById('steps-value');
const resolutionValueSpan = document.getElementById('resolution-value');
const fpsSpan = document.getElementById('fps');
const gpuInfoSpan = document.getElementById('gpu-info');
const renderStatusSpan = document.getElementById('render-status');
const gpuModelSpan = document.getElementById('gpu-model-name');
const statusIndicator = document.querySelector('.status-indicator');

let currentBackend = null;
let animationFrameId = null;
let startTime = performance.now();

const settings = {
    scene : 'crystal',
    bounces : 3,
    shadowQuality : 24,
    maxSteps : 120,
    resolutionScale : 1.0
};

function setBackendCookie(backendName, days = 7) {
    const d = new Date();
    d.setTime(d.getTime() + days * 24 * 60 * 60 * 1000);
    document.cookie = `gpuBackend=${encodeURIComponent(backendName)};expires=${d.toUTCString()};path=/;SameSite=Lax`;
}
function getBackendCookie() {
    const match = document.cookie.match(/(?:^|; )gpuBackend=([^;]*)/);
    return match ? decodeURIComponent(match[1]) : null;
}
function clearBackendCookie() { document.cookie = 'gpuBackend=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/'; }

window.addEventListener('load', async () => {
    // 尽量在任何获取 canvas context 之前读取 cookie 并设定后端选择
    const backendName = getBackendCookie() || 'webgpu'; // 默认 webgpu
    backendSelector.value = backendName;
    onSettingsChange();
    // 调用你现有的 switchBackend（注意：switchBackend 内不应该再重复设置 cookie 否则会循环）
    try {
        await switchBackend(backendName);
    }
    catch (err) {
        console.error('初始化后端失败：', err);
        // 处理失败：降级到 webgl 或展示错误
    }
});

// --- 初始化画布 ---
function setupCanvas() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, canvas.clientWidth * dpr * settings.resolutionScale);
    canvas.height = Math.max(1, canvas.clientHeight * dpr * settings.resolutionScale);
}

setupCanvas();
window.addEventListener('resize', setupCanvas);

// --- FPS 计数器 ---
let lastFrameTime = 0;
let frameCount = 0;
function updateFPS(now) {
    frameCount++;
    if (now - lastFrameTime >= 1000) {
        const fps = Math.min(frameCount, 120);
        fpsSpan.textContent = fps;

        // 更新性能状态
        if (fps > 60) {
            renderStatusSpan.textContent = "性能极佳";
            renderStatusSpan.style.color = "#00e676";
            statusIndicator.className = "status-indicator status-optimal";
        } else if (fps > 30) {
            renderStatusSpan.textContent = "性能良好";
            renderStatusSpan.style.color = "#ffea00";
            statusIndicator.className = "status-indicator status-good";
        } else {
            renderStatusSpan.textContent = "负载过高";
            renderStatusSpan.style.color = "#ff5252";
            statusIndicator.className = "status-indicator status-high";
        }

        frameCount = 0;
        lastFrameTime = now;
    }
}

// --- WebGL 后端 ---
const webglBackend = {
    gl : null,
    program : null,
    positionBuffer : null,
    vao : null,

    init() {
        this.gl = canvas.getContext('webgl2');
        if (!this.gl) {
            console.error('WebGL2 not supported!');
            return false;
        }

        const debugInfo = this.gl.getExtension('WEBGL_debug_renderer_info');
        const gpuInfo = debugInfo ? this.gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'GPU';
        gpuInfoSpan.textContent = gpuInfo;
        gpuModelSpan.textContent = gpuInfo;

        const vsSource = `#version 300 es
                    in vec2 a_position;
                    void main() { gl_Position = vec4(a_position, 0.0, 1.0); }
                `;

        // 动态生成着色器
        const selectedScene = scenes[settings.scene] || scenes.crystal;
        let shaderCode = shaderCore.replace('{{MAP_FUNCTION}}', selectedScene.map)
                             .replace('{{COLOR_FUNCTION}}', selectedScene.getColor);

        const fsSource = `#version 300 es
                    precision highp float;
                    uniform vec2 u_resolution;
                    uniform float u_time;
                    uniform int u_bounces;
                    uniform int u_shadow_quality;
                    uniform int u_max_steps;

                    out vec4 outColor;

                    ${shaderCode}

                    vec3 render(vec3 ro, vec3 rd) {
                        vec3 final_color = vec3(0.0);
                        vec3 accum = vec3(1.0);

                        for (int i = 0; i < 10; i++) {
                            if (i > u_bounces) break;

                            float t = 0.0;
                            vec3 hit_pos;
                            vec4 scene_info;

                            for (int j = 0; j < 200; j++) {
                                if (j >= u_max_steps) break;
                                hit_pos = ro + rd * t;
                                scene_info = map(hit_pos);
                                if (scene_info.x < 0.001 || t > 100.0) {
                                    break;
                                }
                                t += scene_info.x;
                            }

                            if (scene_info.x < 0.001) {
                                vec3 normal = get_normal(hit_pos);
                                vec3 light_pos = vec3(2.0, 5.0, -3.0);
                                vec3 light_dir = normalize(light_pos - hit_pos);
                                float diffuse = max(0.0, dot(normal, light_dir));

                                float shadow = soft_shadow(hit_pos + normal * 0.01, light_dir, 10.0, float(u_shadow_quality), u_shadow_quality);
                                diffuse *= shadow;

                                vec3 base_color = get_color(scene_info.y);
                                vec3 emission = get_emission(scene_info.y, scene_info.w);

                                final_color += accum * (base_color * (diffuse + 0.1) + emission);

                                // 反射
                                if (scene_info.z > 0.0) {
                                    accum *= scene_info.z * base_color;
                                    ro = hit_pos + normal * 0.01;
                                    rd = reflect(rd, normal);
                                } else {
                                    break;
                                }
                            } else {
                                // 天空颜色
                                vec3 sky_color = vec3(0.05, 0.08, 0.12);
                                sky_color = mix(sky_color, vec3(0.3, 0.5, 0.8), smoothstep(0.0, 0.5, rd.y));
                                final_color += accum * sky_color;
                                break;
                            }
                        }
                        return final_color;
                    }

                    void main() {
                        vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;

                        // 相机动画
                        ${selectedScene.cameraPath}
                        vec3 look_at = vec3(0.0, 0.0, 0.0);

                        vec3 f = normalize(look_at - ro);
                        vec3 r = normalize(cross(vec3(0,1,0), f));
                        vec3 u = cross(f, r);
                        vec3 rd = normalize(f + uv.x * r + uv.y * u);

                        vec3 color = render(ro, rd);

                        // 色调映射和伽马校正
                        color = pow(color, vec3(0.4545));
                        outColor = vec4(color, 1.0);
                    }
                `;

        this.program = this.createProgram(vsSource, fsSource);
        if (!this.program) return false;

        this.positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([ -1, -1, 1, -1, -1, 1, 1, 1 ]), this.gl.STATIC_DRAW);

        this.vao = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.vao);
        const posLoc = this.gl.getAttribLocation(this.program, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        return true;
    },

    createShader(type, source) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader error: ' + this.gl.getShaderInfoLog(shader));
            this.gl.deleteShader(shader);
            return null;
        }
        return shader;
    },

    createProgram(vs, fs) {
        const p = this.gl.createProgram();
        const vsShader = this.createShader(this.gl.VERTEX_SHADER, vs);
        const fsShader = this.createShader(this.gl.FRAGMENT_SHADER, fs);

        if (!vsShader || !fsShader) return null;

        this.gl.attachShader(p, vsShader);
        this.gl.attachShader(p, fsShader);
        this.gl.linkProgram(p);

        if (!this.gl.getProgramParameter(p, this.gl.LINK_STATUS)) {
            console.error('Program link error: ' + this.gl.getProgramInfoLog(p));
            return null;
        }
        return p;
    },

    render(time) {
        this.gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);
        this.gl.useProgram(this.program);
        this.gl.bindVertexArray(this.vao);
        this.gl.uniform2f(this.gl.getUniformLocation(this.program, 'u_resolution'), this.gl.canvas.width,
                          this.gl.canvas.height);
        this.gl.uniform1f(this.gl.getUniformLocation(this.program, 'u_time'), time * 0.001);
        this.gl.uniform1i(this.gl.getUniformLocation(this.program, 'u_bounces'), settings.bounces);
        this.gl.uniform1i(this.gl.getUniformLocation(this.program, 'u_shadow_quality'), settings.shadowQuality);
        this.gl.uniform1i(this.gl.getUniformLocation(this.program, 'u_max_steps'), settings.maxSteps);
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    },

    cleanup() {
        if (this.gl) {
            if (this.program) this.gl.deleteProgram(this.program);
            if (this.positionBuffer) this.gl.deleteBuffer(this.positionBuffer);
            if (this.vao) this.gl.deleteVertexArray(this.vao);
        }
    }
};

// --- WebGPU 后端 ---
const webgpuBackend = {
    device : null,
    context : null,
    pipeline : null,
    uniformBuffer : null,
    bindGroup : null,
    async init() {
        if (!navigator.gpu) {
            console.error('WebGPU not supported!');
            return false;
        }
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.error('No WebGPU adapter found!');
                return false;
            }
            const gpuInfo = 'GPU';
            gpuInfoSpan.textContent = gpuInfo;
            gpuModelSpan.textContent = gpuInfo;
            this.device = await adapter.requestDevice();
            this.context = canvas.getContext('webgpu');
            const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
            this.context.configure({device : this.device, format : presentationFormat, alphaMode : 'opaque'});
            // 使用WGSL语法重写着色器
            const wgslShader = `
                struct Uniforms {
                    resolution: vec2f,
                    time: f32,
                    bounces: i32,
                    shadow_quality: i32,
                    max_steps: i32,
                };
                @group(0) @binding(0) var<uniform> u: Uniforms;
                // 基础形状函数
                fn sdSphere(p: vec3f, s: f32) -> f32 {
                    return length(p) - s;
                }
                fn sdPlane(p: vec3f) -> f32 {
                    return p.y;
                }
                fn sdBox(p: vec3f, b: vec3f) -> f32 {
                    let q = abs(p) - b;
                    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
                }
                fn sdTorus(p: vec3f, t: vec2f) -> f32 {
                    let q = vec2f(length(p.xz) - t.x, p.y);
                    return length(q) - t.y;
                }
                fn sdCylinder(p: vec3f, r: f32, h: f32) -> f32 {
                    let d = abs(vec2f(length(p.xz), p.y)) - vec2f(r, h);
                    return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
                }
                fn sdCapsule(p: vec3f, h: f32, r: f32) -> f32 {
                    let adjusted_p = vec3f(p.x, p.y - clamp(p.y, 0.0, h), p.z);
                    return length(adjusted_p) - r;
                }
                fn sdCone(p: vec3f, c: vec2f) -> f32 {
                    let q = length(p.xz);
                    return dot(c, vec2f(q, p.y));
                }
                // 使用%运算符替代mod函数
                fn opRep(p: vec3f, c: vec3f) -> vec3f {
                    return p % c - 0.5 * c;
                }
                fn opUnion(a: vec4f, b: vec4f) -> vec4f {
                    if (a.x < b.x) {
                        return a;
                    }
                    return b;
                }
                // 水晶宫殿场景
                fn map(p: vec3f) -> vec4f {
                    let p_rep = opRep(p, vec3f(4.0, 4.0, 4.0));

                    // 主水晶结构
                    let mainCrystal = vec4f(sdSphere(p_rep - vec3f(0.0, 0.0, 0.0), 0.8), 1.0, 0.95, 0.0);

                    // 地面
                    let ground = vec4f(sdPlane(p), 2.0, 0.5, 0.0);

                    // 辅助水晶结构
                    let crystal1 = vec4f(sdSphere(p_rep - vec3f(1.5, 0.5, 1.5), 0.5), 1.0, 0.92, 0.0);
                    let crystal2 = vec4f(sdSphere(p_rep - vec3f(-1.5, 0.7, -1.5), 0.6), 1.0, 0.9, 0.0);

                    // 柱子
                    let pillar1 = vec4f(sdCylinder(p - vec3f(2.0, 0.0, 2.0), 0.3, 2.0), 3.0, 0.8, 0.0);
                    let pillar2 = vec4f(sdCylinder(p - vec3f(-2.0, 0.0, -2.0), 0.3, 2.0), 3.0, 0.8, 0.0);

                    var res = ground;
                    res = opUnion(res, mainCrystal);
                    res = opUnion(res, crystal1);
                    res = opUnion(res, crystal2);
                    res = opUnion(res, pillar1);
                    res = opUnion(res, pillar2);

                    return res;
                }

                fn get_normal(p: vec3f) -> vec3f {
                    let e = vec2f(0.001, 0.0);
                    let n = vec3f(
                        map(p + e.xyy).x - map(p - e.xyy).x,
                        map(p + e.yxy).x - map(p - e.yxy).x,
                        map(p + e.yyx).x - map(p - e.yyx).x
                    );
                    return normalize(n);
                }
                fn get_color(mat_id: f32) -> vec3f {
                    if (mat_id < 1.5) {
                        // 水晶材质
                        return vec3f(0.9, 0.95, 1.0);
                    } else if (mat_id < 2.5) {
                        // 地面材质
                        return vec3f(0.7, 0.7, 0.8);
                    }
                    // 柱子材质
                    return vec3f(0.4, 0.6, 0.9);
                }
                fn soft_shadow(ro: vec3f, rd: vec3f, max_dist: f32, k: f32, steps: i32) -> f32 {
                    var res = 1.0;
                    var t = 0.01;
                    for (var i = 0; i < steps; i++) {
                        if (t >= max_dist) { break; }
                        let h = map(ro + rd * t).x;
                        if (h < 0.001) { return 0.0; }
                        res = min(res, k * h / t);
                        t = t + h;
                    }
                    return res;
                }
                fn get_emission(mat_id: f32, glow: f32) -> vec3f {
                    if (glow > 0.0) {
                        if (mat_id < 1.5) {
                            return vec3f(0.2, 0.8, 0.3) * 3.0;
                        }
                        return vec3f(0.8, 0.2, 0.1) * 2.0;
                    }
                    return vec3f(0.0);
                }
                fn render(ro: vec3f, rd: vec3f) -> vec3f {
                    var local_ro: vec3f = ro;            // 可变副本
                    var local_rd: vec3f = rd;            // 可变副本

                    var final_color: vec3f = vec3f(0.0);
                    var accum: vec3f = vec3f(1.0);

                    for (var i: i32 = 0; i < 10; i = i + 1) {
                        if (i >= i32(u.bounces)) { break; }

                        var t: f32 = 0.0;
                        var hit_pos: vec3f = vec3f(0.0);
                        var scene_info: vec4f = vec4f(0.0);

                        for (var j: i32 = 0; j < 200; j = j + 1) {
                            if (j >= i32(u.max_steps)) { break; }
                            hit_pos = local_ro + local_rd * t;
                            scene_info = map(hit_pos);
                            if (scene_info.x < 0.001 || t > 100.0) {
                                break;
                            }
                            t = t + scene_info.x;
                        }

                        if (scene_info.x < 0.001) {
                            let normal = get_normal(hit_pos);
                            let light_pos = vec3f(2.0, 5.0, -3.0);
                            let light_dir = normalize(light_pos - hit_pos);
                            var diffuse: f32 = max(0.0, dot(normal, light_dir));
                            let shadow = soft_shadow(hit_pos + normal * 0.01, light_dir, 10.0, f32(u.shadow_quality), u.shadow_quality);
                            diffuse = diffuse * shadow;
                            let base_color = get_color(scene_info.y);
                            let emission = get_emission(scene_info.y, scene_info.w);
                            final_color = final_color + accum * (base_color * (diffuse + 0.1) + emission);

                            // 反射：改变 local_ro/local_rd（而不是参数）
                            if (scene_info.z > 0.0) {
                                accum = accum * scene_info.z * base_color;
                                local_ro = hit_pos + normal * 0.01;
                                local_rd = reflect(local_rd, normal);
                            } else {
                                break;
                            }
                        } else {
                            // 天空颜色
                            var sky_color: vec3f = vec3f(0.05, 0.08, 0.12);
                            sky_color = mix(sky_color, vec3f(0.3, 0.5, 0.8), smoothstep(0.0, 0.5, local_rd.y));
                            final_color = final_color + accum * sky_color;
                            break;
                        }
                    }

                    return final_color;
                }

                @vertex
                fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4f {
                    let pos = array<vec2f, 4>(
                        vec2f(-1.0, -1.0), vec2f(1.0, -1.0),
                        vec2f(-1.0, 1.0), vec2f(1.0, 1.0)
                    );
                    return vec4f(pos[in_vertex_index], 0.0, 1.0);
                }
                @fragment
                fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
                    let uv = (frag_coord.xy - 0.5 * u.resolution.xy) / u.resolution.y;
                    // 相机动画
                    let time = u.time * 0.1;
                    var ro = vec3f(5.0 * cos(time), 3.0, 5.0 * sin(time));
                    let look_at = vec3f(0.0, 0.0, 0.0);
                    let f = normalize(look_at - ro);
                    let r = normalize(cross(vec3f(0.0, 1.0, 0.0), f));
                    let u_vec = cross(f, r);
                    let rd = normalize(f + uv.x * r + uv.y * u_vec);
                    let color = render(ro, rd);
                    // 色调映射和伽马校正
                    let final_color = pow(color, vec3f(0.4545));
                    return vec4f(final_color, 1.0);
                }
            `;
            const shaderModule = this.device.createShaderModule({label : 'Ray Tracing Shader', code : wgslShader});
            this.pipeline = this.device.createRenderPipeline({
                label : 'Ray Tracing Pipeline',
                layout : 'auto',
                vertex : {module : shaderModule, entryPoint : 'vs_main'},
                fragment : {module : shaderModule, entryPoint : 'fs_main', targets : [ {format : presentationFormat} ]},
                primitive : {topology : 'triangle-strip'}
            });
            this.uniformBuffer = this.device.createBuffer({
                label : 'Uniform Buffer',
                size : 6 * 4, // 6个32位值
                usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this.bindGroup = this.device.createBindGroup({
                label : 'Bind Group',
                layout : this.pipeline.getBindGroupLayout(0),
                entries : [ {binding : 0, resource : {buffer : this.uniformBuffer}} ]
            });
            return true;
        }
        catch (error) {
            console.error('WebGPU initialization error:', error);
            return false;
        }
    },

    render(time) {
        if (!this.device) return;

        const uniformData = new Float32Array(6);
        const uniformDataAsInt = new Int32Array(uniformData.buffer);
        uniformData[0] = canvas.width;
        uniformData[1] = canvas.height;
        uniformData[2] = time * 0.001;
        uniformDataAsInt[3] = settings.bounces;
        uniformDataAsInt[4] = settings.shadowQuality;
        uniformDataAsInt[5] = settings.maxSteps;

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();

        const passEncoder = commandEncoder.beginRenderPass({
            colorAttachments : [ {
                view : textureView,
                clearValue : {r : 0.05, g : 0.08, b : 0.12, a : 1.0},
                loadOp : 'clear',
                storeOp : 'store'
            } ]
        });

        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.bindGroup);
        passEncoder.draw(4);
        passEncoder.end();

        this.device.queue.submit([ commandEncoder.finish() ]);
    },

    cleanup() {
        // WebGPU资源会自动垃圾回收
    }
};

// --- 场景定义 ---
// --- 场景定义（修复括号/分号） ---
const scenes = {
    crystal : {
        map : `
            vec4 map(vec3 p) {
                // 水晶宫殿场景
                vec3 p_rep = opRep(p, vec3(4.0, 4.0, 4.0));

                // 主水晶结构
                vec4 mainCrystal = vec4(sdSphere(p_rep - vec3(0.0, 0.0, 0.0), 0.8), 1.0, 0.95, 0.0);

                // 地面
                vec4 ground = vec4(sdPlane(p), 2.0, 0.5, 0.0);

                // 辅助水晶结构
                vec4 crystal1 = vec4(sdSphere(p_rep - vec3(1.5, 0.5, 1.5), 0.5), 1.0, 0.92, 0.0);
                vec4 crystal2 = vec4(sdSphere(p_rep - vec3(-1.5, 0.7, -1.5), 0.6), 1.0, 0.9, 0.0);

                // 柱子
                vec4 pillar1 = vec4(sdCylinder(p - vec3(2.0, 0.0, 2.0), 0.3, 2.0), 3.0, 0.8, 0.0);
                vec4 pillar2 = vec4(sdCylinder(p - vec3(-2.0, 0.0, -2.0), 0.3, 2.0), 3.0, 0.8, 0.0);

                vec4 res = ground;
                res = opUnion(res, mainCrystal);
                res = opUnion(res, crystal1);
                res = opUnion(res, crystal2);
                res = opUnion(res, pillar1);
                res = opUnion(res, pillar2);

                return res;
            }
        `,
        getColor : `
            vec3 get_color(float mat_id) {
                if (mat_id < 1.5) {
                    // 水晶材质
                    return vec3(0.9, 0.95, 1.0);
                }
                if (mat_id < 2.5) {
                    // 地面材质
                    return vec3(0.7, 0.7, 0.8);
                }
                // 柱子材质
                return vec3(0.4, 0.6, 0.9);
            }
        `,
        cameraPath : `
            float time = u_time * 0.1;
            vec3 ro = vec3(5.0 * cos(time), 3.0, 5.0 * sin(time));
        `
    },
    future : {
        map : `
            vec4 map(vec3 p) {
                // 未来城市场景
                vec3 p_rep = opRep(p, vec3(6.0, 10.0, 6.0));

                // 建筑主体
                vec4 building1 = vec4(sdBox(p_rep - vec3(0.0, 2.0, 0.0), vec3(1.0, 3.0, 1.0)), 1.0, 0.85, 0.0);
                vec4 building2 = vec4(sdBox(p_rep - vec3(2.0, 3.0, 2.0), vec3(0.8, 4.0, 0.8)), 1.0, 0.8, 0.0);
                vec4 building3 = vec4(sdBox(p_rep - vec3(-2.0, 4.0, -2.0), vec3(0.6, 5.0, 0.6)), 1.0, 0.75, 0.0);

                // 地面
                vec4 ground = vec4(sdPlane(p), 2.0, 0.5, 0.0);

                // 飞行器
                vec4 aircraft = vec4(sdCapsule(p - vec3(0.0, 5.0 + sin(u_time * 0.5), 0.0), 1.0, 0.5), 3.0, 0.9, 0.0);

                vec4 res = ground;
                res = opUnion(res, building1);
                res = opUnion(res, building2);
                res = opUnion(res, building3);
                res = opUnion(res, aircraft);

                return res;
            }
        `,
        getColor : `
            vec3 get_color(float mat_id) {
                if (mat_id < 1.5) {
                    // 建筑材质
                    return vec3(0.9, 0.3, 0.2);
                }
                if (mat_id < 2.5) {
                    // 地面材质
                    return vec3(0.2, 0.2, 0.3);
                }
                // 飞行器材质
                return vec3(0.0, 0.8, 1.0);
            }
        `,
        cameraPath : `
            float time = u_time * 0.15;
            vec3 ro = vec3(8.0 * cos(time), 4.0, 8.0 * sin(time));
        `
    }
};

const shaderCore = `
            // 基础形状函数
            float sdSphere(vec3 p, float s) {
                return length(p) - s;
            }

            float sdPlane(vec3 p) {
                return p.y;
            }

            float sdBox(vec3 p, vec3 b) {
                vec3 q = abs(p) - b;
                return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
            }

            float sdTorus(vec3 p, vec2 t) {
                vec2 q = vec2(length(p.xz)-t.x,p.y);
                return length(q)-t.y;
            }

            float sdCylinder(vec3 p, float r, float h) {
                vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
                return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
            }

            float sdCapsule(vec3 p, float h, float r) {
                p.y -= clamp(p.y, 0.0, h);
                return length(p) - r;
            }

            float sdCone(vec3 p, vec2 c) {
                float q = length(p.xz);
                return dot(c, vec2(q, p.y));
            }

            vec3 opRep(vec3 p, vec3 c) {
                return mod(p, c) - 0.5 * c;
            }

            vec4 opUnion(vec4 a, vec4 b) {
                return a.x < b.x ? a : b;
            }

            // 场景映射函数
            {{MAP_FUNCTION}}

            vec3 get_normal(vec3 p) {
                vec2 e = vec2(0.001, 0.0);
                vec3 n = vec3(
                    map(p + e.xyy).x - map(p - e.xyy).x,
                    map(p + e.yxy).x - map(p - e.yxy).x,
                    map(p + e.yyx).x - map(p - e.yyx).x
                );
                return normalize(n);
            }

            // 材质颜色函数
            {{COLOR_FUNCTION}}

            float soft_shadow(vec3 ro, vec3 rd, float max_dist, float k, int steps) {
                float res = 1.0;
                float t = 0.01;
                for (int i = 0; i < steps; i++) {
                    if (t >= max_dist) { break; }
                    float h = map(ro + rd * t).x;
                    if (h < 0.001) { return 0.0; }
                    res = min(res, k * h / t);
                    t = t + h;
                }
                return res;
            }

            // 发光效果
            vec3 get_emission(float mat_id, float glow) {
                if (glow > 0.0) {
                    if (mat_id < 1.5) { return vec3(0.2, 0.8, 0.3) * 3.0; }
                    return vec3(0.8, 0.2, 0.1) * 2.0;
                }
                return vec3(0.0);
            }
        `;

// --- 主逻辑 ---
async function switchBackend(backendName) {

    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    if (currentBackend?.cleanup) currentBackend.cleanup();

    loadingScreen.style.display = 'flex';
    loadingScreen.style.opacity = '1';
    renderStatusSpan.textContent = "初始化中...";
    statusIndicator.className = "status-indicator status-good";

    setupCanvas();

    try {
        currentBackend = backendName === 'webgl' ? webglBackend : webgpuBackend;
        const success = await (currentBackend === webgpuBackend ? webgpuBackend.init() : webglBackend.init());

        if (success) {
            lastFrameTime = performance.now();
            frameCount = 0;
            startTime = lastFrameTime;
            renderStatusSpan.textContent = "渲染中...";
            statusIndicator.className = "status-indicator status-optimal";
            animate(lastFrameTime);

            // 隐藏加载界面
            setTimeout(() => {
                loadingScreen.style.opacity = '0';
                setTimeout(() => { loadingScreen.style.display = 'none'; }, 500);
            }, 500);
        } else {
            renderStatusSpan.textContent = "初始化失败";
            statusIndicator.className = "status-indicator status-high";
            loadingScreen.style.display = 'none';
        }
    }
    catch (error) {
        console.error("后端初始化错误:", error);
        renderStatusSpan.textContent = "错误: " + error.message;
        statusIndicator.className = "status-indicator status-high";
        loadingScreen.style.display = 'none';
    }
}

function animate(now) {
    updateFPS(now);

    try {
        if (currentBackend && currentBackend.render) {
            currentBackend.render(now - startTime);
        }
    }
    catch (error) {
        console.error("渲染错误:", error);
        renderStatusSpan.textContent = "渲染错误: " + error.message;
        statusIndicator.className = "status-indicator status-high";
        cancelAnimationFrame(animationFrameId);
        return;
    }

    animationFrameId = requestAnimationFrame(animate);
}

function onSettingsChange() {
    settings.bounces = parseInt(bouncesSlider.value);
    settings.shadowQuality = parseInt(shadowSlider.value);
    settings.maxSteps = parseInt(stepsSlider.value);
    settings.resolutionScale = parseInt(resolutionSlider.value) / 100;

    bouncesValueSpan.textContent = settings.bounces;
    shadowValueSpan.textContent = settings.shadowQuality;
    stepsValueSpan.textContent = settings.maxSteps;
    resolutionValueSpan.textContent = parseInt(resolutionSlider.value) + "%";

    setupCanvas();

    // 重新初始化后端以应用新设置
    if (currentBackend) {
        switchBackend(backendSelector.value);
    }
}

function onShaderChange() {
    settings.scene = shaderSelector.value;
    if (currentBackend) {
        switchBackend(backendSelector.value);
    }
}

function onShaderPreset(e) {
    shaderPresets.forEach(btn => btn.classList.remove('active'));
    e.target.classList.add('active');

    if (e.target.dataset.preset === 'extreme') {
        bouncesSlider.value = 6;
        shadowSlider.value = 48;
        stepsSlider.value = 180;
        resolutionSlider.value = 120;
    } else {
        bouncesSlider.value = 3;
        shadowSlider.value = 24;
        stepsSlider.value = 120;
        resolutionSlider.value = 100;
    }

    onSettingsChange();
}

function onSceneSelect(e) {
    const scene = e.currentTarget.dataset.scene;
    settings.scene = scene;

    previewItems.forEach(item => item.classList.remove('active'));
    e.currentTarget.classList.add('active');

    shaderSelector.value = scene;

    if (currentBackend) {
        switchBackend(backendSelector.value);
    }
}

// 事件监听器
backendSelector.addEventListener('change', (e) => {
    const newBackend = e.target.value;
    // 记录用户选择（7天有效）
    setBackendCookie(newBackend, 7);

    // 如果你想在刷新前显示提示，可以短暂显示 UI，下面直接刷新
    // 立即刷新页面以确保新的上下文在“干净”的 canvas 上被创建
    location.reload();
});
shaderSelector.addEventListener('change', onShaderChange);
bouncesSlider.addEventListener('input', onSettingsChange);
shadowSlider.addEventListener('input', onSettingsChange);
stepsSlider.addEventListener('input', onSettingsChange);
resolutionSlider.addEventListener('input', onSettingsChange);

shaderPresets.forEach(btn => { btn.addEventListener('click', onShaderPreset); });

previewItems.forEach(item => { item.addEventListener('click', onSceneSelect); });

// 初始设置
// onSettingsChange();
// switchBackend(backendSelector.value);

// 初始GPU信息
if (navigator.gpu) {
    navigator.gpu.requestAdapter().then(adapter => {
        if (adapter) {
            // adapter.requestAdapterInfo().then(info => {
            gpuModelSpan.textContent = /*info.description ||*/ 'GPU';
            //});
        }
    });
}
