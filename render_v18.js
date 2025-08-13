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
    const backendName = getBackendCookie() || 'webgpu';
    backendSelector.value = backendName;
    onSettingsChange();

    try {
        if (!isPaused) {
            await switchBackend(backendName);
        }
    }
    catch (err) {
        console.error('初始化后端失败：', err);
        renderStatusSpan.textContent = "初始化失败";
        statusIndicator.className = "status-indicator status-high";
    }
});

// --- 初始化画布 ---
function setupCanvas() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, 1000 * settings.resolutionScale);
    canvas.height = Math.max(1, 800 * settings.resolutionScale);
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
        const gpuInfo = debugInfo ? this.gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'WebGL';
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
            const gpuInfo = 'WebGPU';
            gpuInfoSpan.textContent = gpuInfo;
            gpuModelSpan.textContent = gpuInfo;
            this.device = await adapter.requestDevice();
            this.context = canvas.getContext('webgpu');
            const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
            this.context.configure({device : this.device, format : presentationFormat, alphaMode : 'opaque'});

            // 获取当前场景的WGSL代码
            const sceneWGSL = wgslScenes[settings.scene] || wgslScenes.crystal;

            // 主WGSL着色器代码
            const wgslShader = `
                struct Uniforms {
                    resolution: vec2f,
                    time: f32,
                    bounces: i32,
                    shadow_quality: i32,
                    max_steps: i32,
                };
                @group(0) @binding(0) var<uniform> u: Uniforms;
                fn sdOctahedron(p: vec3f, s: f32) -> f32 {
                    // 简化版的八面体 SDF
                    let m = abs(p.x) + abs(p.y) + abs(p.z) - s;
                    return m;
                }
                        
                fn sdEllipsoid(p: vec3f, r: vec3f) -> f32 {
                    // 椭球 SDF
                    return length(p / r) - 1.0;
                }

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
                
                // 场景特定函数
                ${sceneWGSL.map}
                
                fn get_normal(p: vec3f) -> vec3f {
                    let e = vec2f(0.001, 0.0);
                    let n = vec3f(
                        map(p + e.xyy).x - map(p - e.xyy).x,
                        map(p + e.yxy).x - map(p - e.yxy).x,
                        map(p + e.yyx).x - map(p - e.yyx).x
                    );
                    return normalize(n);
                }
                
                ${sceneWGSL.getColor}
                
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
                
                ${sceneWGSL.getEmission}
                
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
                    return vec4f(pos[in_vertex_index].x, -pos[in_vertex_index].y, 0.0, 1.0);
                }
                
                @fragment
                fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
                    let uv = (frag_coord.xy - 0.5 * u.resolution.xy) / u.resolution.y;
                    
                    // 相机动画
                    ${sceneWGSL.cameraPath}
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
                size : 6 * 4, // 24 bytes (6个32位值)
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

        // 使用 DataView 确保正确的数据类型
        const uniformBufferSize = 6 * 4; // 6个32位值 = 24字节
        const uniformData = new ArrayBuffer(uniformBufferSize);
        const uniformDataView = new DataView(uniformData);

        uniformDataView.setFloat32(0, canvas.width, true);  // u_resolution.x
        uniformDataView.setFloat32(4, canvas.height, true); // u_resolution.y
        uniformDataView.setFloat32(8, time * 0.001, true);  // u_time

        // 整数部分使用 setInt32
        uniformDataView.setInt32(12, settings.bounces, true);       // u_bounces
        uniformDataView.setInt32(16, settings.shadowQuality, true); // u_shadow_quality
        uniformDataView.setInt32(20, settings.maxSteps, true);      // u_max_steps

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
        if (this.device) {
            this.device.destroy();
            this.device = null;
        }
        this.context = null;
        this.pipeline = null;
        this.uniformBuffer = null;
        this.bindGroup = null;
    }
};

// --- WGSL 场景定义 ---
const wgslScenes = {
    crystal : {
        map : `
            fn map(p: vec3f) -> vec4f {
                // 水晶宫殿场景
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
        `,
        getColor : `
            fn get_color(mat_id: f32) -> vec3f {
                if (mat_id < 1.5) {
                    // 水晶材质
                    return vec3f(0.9, 0.95, 1.0);
                }
                if (mat_id < 2.5) {
                    // 地面材质
                    return vec3f(0.7, 0.7, 0.8);
                }
                // 柱子材质
                return vec3f(0.4, 0.6, 0.9);
            }
        `,
        getEmission : `
            fn get_emission(mat_id: f32, glow: f32) -> vec3f {
                if (glow > 0.0) {
                    if (mat_id < 1.5) {
                        return vec3f(0.2, 0.8, 0.3) * 3.0;
                    }
                    return vec3f(0.8, 0.2, 0.1) * 2.0;
                }
                return vec3f(0.0);
            }
        `,
        cameraPath : `
            let time = u.time * 0.1;
            var ro = vec3f(5.0 * cos(time), 3.0, 5.0 * sin(time));
        `
    },
    cave : {
        map : `
        fn map(p: vec3f) -> vec4f {
            // 地面
            let ground = vec4f(sdPlane(p), 1.0, 0.5, 0.0);
            
            // 石笋阵列 - 使用传入的p参数
            let stalagmite1 = vec4f(sdCone(p - vec3f(2.0, 0.0, 1.0), vec2f(0.5, 2.0)), 2.0, 0.8, 0.0);
            let stalagmite2 = vec4f(sdCone(p - vec3f(-3.0, 0.0, -2.0), vec2f(0.7, 2.5)), 2.0, 0.8, 0.0);
            let stalagmite3 = vec4f(sdCone(p - vec3f(4.0, 0.0, -3.0), vec2f(0.4, 1.8)), 2.0, 0.8, 0.0);
            
            // 钟乳石阵列 - 使用传入的p参数
            let stalactite1 = vec4f(sdCone(p - vec3f(0.0, 5.0, 0.0), vec2f(0.6, 2.0)), 3.0, 0.8, 0.0);
            let stalactite2 = vec4f(sdCone(p - vec3f(3.0, 4.5, -2.0), vec2f(0.5, 1.5)), 3.0, 0.8, 0.0);
            let stalactite3 = vec4f(sdCone(p - vec3f(-4.0, 4.8, 3.0), vec2f(0.7, 1.8)), 3.0, 0.8, 0.0);
            
            // 水晶簇 - 使用传入的p参数
            let crystal1 = vec4f(sdOctahedron(p - vec3f(3.0, 0.5, 0.0), 0.8), 4.0, 0.95, 0.8);
            let crystal2 = vec4f(sdOctahedron(p - vec3f(3.8, 1.0, 1.5), 0.5), 4.0, 0.95, 0.8);
            let crystal3 = vec4f(sdOctahedron(p - vec3f(2.5, 0.3, -1.0), 0.4), 4.0, 0.95, 0.8);
            
            // 发光水晶柱 - 使用传入的p参数
            let glowColumn = vec4f(sdCylinder(p - vec3f(0.0, 0.5, 0.0), 0.5, 2.0), 5.0, 0.9, 1.5);
            
            var res = ground;
            res = opUnion(res, stalagmite1);
            res = opUnion(res, stalagmite2);
            res = opUnion(res, stalagmite3);
            res = opUnion(res, stalactite1);
            res = opUnion(res, stalactite2);
            res = opUnion(res, stalactite3);
            res = opUnion(res, crystal1);
            res = opUnion(res, crystal2);
            res = opUnion(res, crystal3);
            res = opUnion(res, glowColumn);
            
            return res;
        }
    `,
        getColor : `
        fn get_color(mat_id: f32) -> vec3f {
            if (mat_id < 1.5) {
                return vec3f(0.35, 0.3, 0.25); // 地面
            }
            if (mat_id < 2.5) {
                return vec3f(0.85, 0.8, 0.75); // 石笋
            }
            if (mat_id < 3.5) {
                return vec3f(0.8, 0.78, 0.7);  // 钟乳石
            }
            if (mat_id < 4.5) {
                return vec3f(0.7, 0.9, 1.0);   // 水晶
            }
            return vec3f(0.1, 0.8, 0.6);       // 发光柱
        }
    `,
        getEmission : `
        fn get_emission(mat_id: f32, glow: f32) -> vec3f {
            if (mat_id > 4.5) {
                return vec3f(0.1, 0.9, 0.7) * 6.0 * glow;
            }
            if (mat_id > 3.5 && mat_id < 4.5) {
                return vec3f(0.6, 0.9, 1.0) * 0.8 * glow;
            }
            return vec3f(0.0);
        }
    `,
        cameraPath : `
        let time = u.time * 0.1;
        var ro = vec3f(
            5.0 * cos(time),
            2.5 + sin(time * 0.5),
            5.0 * sin(time)
        );
    `
    },
    space : {
        map : `
        fn map(p: vec3f) -> vec4f {
            // 主环形结构 - 使用传入的p参数
            let ring = vec4f(sdTorus(p, vec2f(4.0, 0.7)), 1.0, 0.85, 0.0);
            
            // 中心球体 - 使用传入的p参数
            let core = vec4f(sdSphere(p, 1.8), 2.0, 0.9, 0.0);
            
            // 太阳能板阵列 - 使用传入的p参数
            let solarPanel1 = vec4f(sdBox(p - vec3f(0.0, 0.0, -6.0), vec3f(5.0, 0.1, 1.0)), 4.0, 0.9, 0.0);
            let solarPanel2 = vec4f(sdBox(p - vec3f(0.0, 0.0, 6.0), vec3f(5.0, 0.1, 1.0)), 4.0, 0.9, 0.0);
            
            // 居住舱 - 使用传入的p参数
            let habitat = vec4f(sdCapsule(p - vec3f(5.0, 0.0, 0.0), 2.0, 1.0), 3.0, 0.7, 0.0);
            
            // 推进器组 - 使用传入的p参数
            let thruster1 = vec4f(sdCone(p - vec3f(-5.0, 0.0, 0.0), vec2f(0.8, 1.8)), 6.0, 0.95, 1.2);
            let thruster2 = vec4f(sdCone(p - vec3f(-5.0, 1.5, 1.5), vec2f(0.4, 1.2)), 6.0, 0.95, 1.2);
            let thruster3 = vec4f(sdCone(p - vec3f(-5.0, -1.5, 1.5), vec2f(0.4, 1.2)), 6.0, 0.95, 1.2);
            
            // 卫星 - 使用传入的p参数
            let satellite = vec4f(sdBox(p - vec3f(3.0, 3.0, 3.0), vec3f(0.5, 0.5, 1.5)), 5.0, 0.8, 0.0);
            
            var res = ring;
            res = opUnion(res, core);
            res = opUnion(res, solarPanel1);
            res = opUnion(res, solarPanel2);
            res = opUnion(res, habitat);
            res = opUnion(res, thruster1);
            res = opUnion(res, thruster2);
            res = opUnion(res, thruster3);
            res = opUnion(res, satellite);
            
            return res;
        }
    `,
        getColor : `
        fn get_color(mat_id: f32) -> vec3f {
            if (mat_id < 1.5) {
                return vec3f(0.7, 0.7, 0.75); // 主环
            }
            if (mat_id < 2.5) {
                return vec3f(0.95, 0.95, 1.0); // 核心
            }
            if (mat_id < 3.5) {
                return vec3f(0.9, 0.92, 0.95); // 居住舱
            }
            if (mat_id < 4.5) {
                return vec3f(0.05, 0.05, 0.2); // 太阳能板
            }
            if (mat_id < 5.5) {
                return vec3f(0.8, 0.8, 0.85); // 卫星
            }
            return vec3f(0.1, 0.3, 0.9);      // 推进器
        }
    `,
        getEmission : `
        fn get_emission(mat_id: f32, glow: f32) -> vec3f {
            if (mat_id > 5.5) {
                return vec3f(0.2, 0.5, 1.0) * 8.0 * glow;
            }
            return vec3f(0.0);
        }
    `,
        cameraPath : `
        let time = u.time * 0.15;
        var ro = vec3f(
            8.0 * cos(time),
            2.0 * sin(time * 0.5),
            8.0 * sin(time)
        );
    `
    },
    future : {
        map : `
        fn sdSkyscraper(p: vec3f, pos: vec3f, width: f32, height: f32) -> vec4f {
            return vec4f(sdBox(p - pos, vec3f(width, height, width)), 2.0, 0.85, 0.0);
        }
        
        fn sdWindow(p: vec3f, pos: vec3f) -> vec4f {
            return vec4f(sdBox(p - pos, vec3f(0.1, 0.5, 0.01)), 6.0, 0.6, 1.0);
        }
        
        fn map(p: vec3f) -> vec4f {
            // 地面网格 - 使用传入的p参数
            let ground = vec4f(sdPlane(p), 1.0, 0.5, 0.0);
            
            // 摩天楼群 - 使用传入的p参数
            let tower1 = sdSkyscraper(p, vec3f(2.0, 5.0, 1.0), 0.8, 8.0);
            let tower2 = sdSkyscraper(p, vec3f(-1.0, 7.0, -2.0), 1.2, 12.0);
            let tower3 = sdSkyscraper(p, vec3f(4.0, 4.0, -3.0), 0.7, 6.0);
            let tower4 = sdSkyscraper(p, vec3f(-3.0, 6.0, 3.0), 1.0, 10.0);
            
            // 悬浮平台 - 使用传入的p参数
            let platform = vec4f(sdBox(p - vec3f(0.0, 12.0, 0.0), vec3f(8.0, 0.5, 8.0)), 3.0, 0.8, 0.0);
            
            // 连接桥 - 使用传入的p参数
            let bridge = vec4f(sdCapsule(p - vec3f(0.0, 13.0, 0.0), 6.0, 0.3), 4.0, 0.7, 0.0);
            
            // 动态飞行器 - 使用传入的p参数
            let aircraft = vec4f(
                sdEllipsoid(p - vec3f(
                    3.0 * sin(u.time * 0.3),
                    10.0 + 2.0 * cos(u.time * 0.7),
                    3.0 * cos(u.time * 0.3)
                ), vec3f(1.5, 0.4, 0.8)),
                5.0, 0.9, 0.0);
            
            // 建筑细节 (窗户) - 使用传入的p参数
            let window1 = sdWindow(p, vec3f(2.0, 8.0, 1.0));
            let window2 = sdWindow(p, vec3f(2.0, 11.0, 1.0));
            
            var res = ground;
            res = opUnion(res, tower1);
            res = opUnion(res, tower2);
            res = opUnion(res, tower3);
            res = opUnion(res, tower4);
            res = opUnion(res, platform);
            res = opUnion(res, bridge);
            res = opUnion(res, aircraft);
            res = opUnion(res, window1);
            res = opUnion(res, window2);
            
            return res;
        }
    `,
        getColor : `
        fn get_color(mat_id: f32) -> vec3f {
            if (mat_id < 1.5) {
                return vec3f(0.1, 0.1, 0.15); // 地面
            }
            if (mat_id < 2.5) {
                return vec3f(0.2, 0.6, 0.9);  // 摩天楼
            }
            if (mat_id < 3.5) {
                return vec3f(0.9, 0.9, 0.95); // 平台
            }
            if (mat_id < 4.5) {
                return vec3f(0.95, 0.7, 0.3); // 桥梁
            }
            if (mat_id < 5.5) {
                return vec3f(0.95, 0.2, 0.1); // 飞行器
            }
            return vec3f(1.0, 0.9, 0.5);      // 窗户
        }
    `,
        getEmission : `
        fn get_emission(mat_id: f32, glow: f32) -> vec3f {
            if (mat_id > 5.5) {
                return mix(vec3f(0.8, 0.7, 0.4), vec3f(1.0, 0.9, 0.6), abs(sin(u.time*2.0))) * 3.0;
            }
            return vec3f(0.0);
        }
    `,
        cameraPath : `
        let time = u.time * 0.2;
        var ro = vec3f(
            10.0 * cos(time * 0.6),
            10.0 + 3.0 * sin(time * 0.4),
            10.0 * sin(time * 0.6)
        );
    `
    }
};

// --- 场景定义 ---
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
    cave : {
        map : `
            vec4 map(vec3 p) {
                // 洞穴墙壁
                vec4 cave = vec4(sdSphere(p, 10.0), 1.0, 0.5, 0.0);
                
                // 石笋 (从地面向上生长)
                vec4 stalagmite = vec4(
                    sdCone(
                        p - vec3(0.0, -8.0, 0.0), 
                        vec2(1.0, 3.0)
                    ), 2.0, 0.8, 0.0);
                
                // 钟乳石 (从洞顶向下生长)
                vec4 stalactite = vec4(
                    sdCone(
                        vec3(p.x, -p.y + 8.0, p.z), 
                        vec2(0.7, 2.5)
                    ), 3.0, 0.8, 0.0);
                
                // 发光水晶
                vec4 glowCrystal = vec4(
                    sdSphere(p - vec3(3.0, -5.0, 2.0), 1.0), 
                    4.0, 0.9, 1.0);
                
                vec4 glowCrystal2 = vec4(
                    sdSphere(p - vec3(-2.0, -6.0, -3.0), 0.8), 
                    4.0, 0.9, 1.0);
                
                vec4 res = cave;
                res = opUnion(res, stalagmite);
                res = opUnion(res, stalactite);
                res = opUnion(res, glowCrystal);
                res = opUnion(res, glowCrystal2);
                
                return res;
            }
        `,
        getColor : `
            vec3 get_color(float mat_id) {
                if (mat_id < 1.5) {
                    // 洞穴墙壁材质
                    return vec3(0.4, 0.3, 0.2);
                }
                if (mat_id < 2.5) {
                    // 石笋材质
                    return vec3(0.8, 0.8, 0.75);
                }
                if (mat_id < 3.5) {
                    // 钟乳石材质
                    return vec3(0.75, 0.75, 0.7);
                }
                // 发光水晶材质
                return vec3(0.2, 0.8, 0.5);
            }
        `,
        cameraPath : `
            float time = u_time * 0.08;
            vec3 ro = vec3(8.0 * cos(time), -2.0, 8.0 * sin(time));
        `
    },
    space : {
        map : `
            vec4 map(vec3 p) {
                // 空间站主体 (环形)
                vec4 ring = vec4(
                    sdTorus(p - vec3(0.0, 0.0, 0.0), vec2(3.0, 0.5)),
                    1.0, 0.85, 0.0);
                
                // 中心球体
                vec4 core = vec4(
                    sdSphere(p - vec3(0.0, 0.0, 0.0), 1.2),
                    1.0, 0.8, 0.0);
                
                // 太阳能板
                vec4 solarPanel1 = vec4(
                    sdBox(p - vec3(0.0, 0.0, -4.5), vec3(4.0, 0.1, 0.5)),
                    2.0, 0.9, 0.0);
                
                vec4 solarPanel2 = vec4(
                    sdBox(p - vec3(0.0, 0.0, 4.5), vec3(4.0, 0.1, 0.5)),
                    2.0, 0.9, 0.0);
                
                // 居住舱
                vec4 habitat = vec4(
                    sdCapsule(p - vec3(3.0, 0.0, 0.0), 1.5, 0.8),
                    3.0, 0.7, 0.0);
                
                // 推进器
                vec4 thruster1 = vec4(
                    sdCone(p - vec3(-3.0, 0.0, 0.0), vec2(0.5, 1.2)),
                    4.0, 0.9, 1.0);
                
                vec4 res = ring;
                res = opUnion(res, core);
                res = opUnion(res, solarPanel1);
                res = opUnion(res, solarPanel2);
                res = opUnion(res, habitat);
                res = opUnion(res, thruster1);
                
                return res;
            }
        `,
        getColor : `
            vec3 get_color(float mat_id) {
                if (mat_id < 1.5) {
                    // 金属灰色
                    return vec3(0.6, 0.6, 0.65);
                }
                if (mat_id < 2.5) {
                    // 太阳能板深蓝色
                    return vec3(0.1, 0.1, 0.3);
                }
                if (mat_id < 3.5) {
                    // 居住舱白色
                    return vec3(0.9, 0.9, 0.95);
                }
                // 推进器蓝色
                return vec3(0.0, 0.5, 1.0);
            }
        `,
        cameraPath : `
            float time = u_time * 0.12;
            vec3 ro = vec3(
                8.0 * cos(time) * sin(time * 0.3),
                2.0 * sin(time * 0.5),
                8.0 * sin(time) * cos(time * 0.3)
            );
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
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    // 正确清理当前后端
    if (currentBackend?.cleanup) {
        currentBackend.cleanup();
    }

    if (!isPaused) {
        loadingScreen.style.display = 'flex';
        loadingScreen.style.opacity = '1';
        renderStatusSpan.textContent = "初始化中...";
        statusIndicator.className = "status-indicator status-good";
    }

    setupCanvas();

    try {
        currentBackend = backendName === 'webgl' ? webglBackend : webgpuBackend;
        const success = await (currentBackend === webgpuBackend ? webgpuBackend.init() : webglBackend.init());

        if (success) {
            lastFrameTime = performance.now();
            frameCount = 0;
            startTime = lastFrameTime;
            renderStatusSpan.textContent = "正在渲染...";
            statusIndicator.className = "status-indicator status-optimal";

            // 只在未暂停时启动渲染循环
            if (!isPaused) {
                startRenderLoop();
            } else {
                loadingScreen.style.display = 'none';
                loadingScreen.style.opacity = '0';
            }

            if (backendName === 'webgl') {
                animate(lastFrameTime);
            }

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
    if (isPaused) return;
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

function onSettingsChange(forceReinit = false) {
    const oldResolution = settings.resolutionScale;
    const oldScene = settings.scene;

    settings.bounces = parseInt(bouncesSlider.value);
    settings.shadowQuality = parseInt(shadowSlider.value);
    settings.maxSteps = parseInt(stepsSlider.value);
    settings.resolutionScale = parseInt(resolutionSlider.value) / 100;

    bouncesValueSpan.textContent = settings.bounces;
    shadowValueSpan.textContent = settings.shadowQuality;
    stepsValueSpan.textContent = settings.maxSteps;
    resolutionValueSpan.textContent = parseInt(resolutionSlider.value) + "%";

    setupCanvas();

    // 只在必要时重新初始化
    const resolutionChanged = oldResolution !== settings.resolutionScale;
    const sceneChanged = oldScene !== settings.scene;

    if ((resolutionChanged || sceneChanged || forceReinit) && currentBackend) {
        switchBackend(backendSelector.value);
    }
}

function onShaderChange() {
    settings.scene = shaderSelector.value;

    // 强制重新初始化以更新着色器
    if (currentBackend) {
        onSettingsChange(true);
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
    setBackendCookie(newBackend, 7);
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
let isPaused = true; // 默认暂停（true）
const pauseBtn = document.getElementById('pause-btn');

function updatePauseButtonUI() {
    if (!pauseBtn) return;
    if (isPaused) {
        pauseBtn.setAttribute('aria-pressed', 'false');
        pauseBtn.classList.remove('active');
        pauseBtn.innerHTML = '<i class="fas fa-play"></i> 开始渲染';
    } else {
        pauseBtn.setAttribute('aria-pressed', 'true');
        pauseBtn.classList.add('active');
        pauseBtn.innerHTML = '<i class="fas fa-pause"></i> 暂停渲染';
    }
}

// 渲染循环控制：开始 / 停止（安全封装）
function startRenderLoop() {
    if (!isPaused && !animationFrameId) {
        // 重置时间基准，避免大跳帧
        startTime = performance.now();
        lastFrameTime = startTime;
        animationFrameId = requestAnimationFrame(animate);
    }
}

function stopRenderLoop() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

// 按钮事件：切换状态（初始化只会改变 isPaused，不会重新 init 后端）
if (pauseBtn) {
    pauseBtn.addEventListener('click', async (e) => {
        isPaused = !isPaused;
        updatePauseButtonUI();

        if (isPaused) {
            // 暂停：停止循环，但保留 currentBackend 实例（可快速恢复）
            stopRenderLoop();
            renderStatusSpan.textContent = "已暂停";
            statusIndicator.className = "status-indicator status-good";
        } else {
            // 恢复：启动循环（如果后端尚未初始化，先初始化）
            renderStatusSpan.textContent = "正在渲染...";
            statusIndicator.className = "status-indicator status-optimal";

            // 如果后端还没 init（比如页面刚 load 但未成功 init），先 init，再启动
            if (!currentBackend) {
                // 使用 selector 的值初始化后端（不改 cookie）
                const desired = backendSelector.value || (getBackendCookie() || 'webgpu');
                try {
                    await switchBackend(desired);
                }
                catch (err) {
                    console.error('恢复时后端初始化失败：', err);
                    renderStatusSpan.textContent = '初始化失败';
                    statusIndicator.className = "status-indicator status-high";
                    return;
                }
            }

            // 启动渲染循环
            startRenderLoop();
        }
    });
}

// 首次页面加载时，保证 UI 显示为“暂停”
updatePauseButtonUI();

// 初始GPU信息
if (navigator.gpu) {
    navigator.gpu.requestAdapter().then(adapter => {
        if (adapter) {
            // adapter.requestAdapterInfo().then(info => {
            gpuModelSpan.textContent = 'WebGPU';
            //});
        }
    });
}
