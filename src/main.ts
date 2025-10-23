const canvas = document.querySelector("canvas");
if (!canvas) {
    throw new Error("Canvas element not found!");
}

const context = canvas.getContext("webgpu");
if (!context) {
    throw new Error("WebGPU context not available!");
}

if (!navigator.gpu) {
    throw new Error("WebGPU is not supported in this browser!");
}

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
}

const device = await adapter.requestDevice();

const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device,
    format: canvasFormat
});

const GRID_SIZE = 32;
const UPDATE_INTERVAL = 200;
const WORKGROUP_SIZE = 8;

const vertices = new Float32Array([
//   X,    Y,
  -0.8, -0.8, // Triangle 1 (Blue)
   0.8, -0.8,
   0.8,  0.8,

  -0.8, -0.8, // Triangle 2 (Red)
   0.8,  0.8,
  -0.8,  0.8,
]);

const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
const cellStateArrayA = new Uint32Array(GRID_SIZE * GRID_SIZE);
const cellStateArrayB = new Uint32Array(GRID_SIZE * GRID_SIZE);

for (let i = 0; i < cellStateArrayA.length; i++) {
    cellStateArrayA[i] = Math.random() > 0.5 ? 1 : 0;
}

for (let i = 0; i < cellStateArrayB.length; i++) {
    cellStateArrayB[i] = i % 2;
}

const vertexBuffer = device.createBuffer({
    label: 'Cell Vertex Buffer',
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
});

const cellStateBuffers = [
    device.createBuffer({
        label: 'Cell State A Buffer',
        size: cellStateArrayA.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    }),
    device.createBuffer({
        label: 'Cell State B Buffer',
        size: cellStateArrayB.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    })
];

const uniformBuffer = device.createBuffer({
    label: 'Grid Uniform Buffer',
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

const cellShaderModule = device.createShaderModule({
    label: 'Cell Shader',
    code: `
        struct VertexInput {
            @location(0) position: vec2f,
            @builtin(instance_index) instance: u32
        };

        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) color: vec2f
        };

        struct FragInput {
            @location(0) color: vec2f
        };

        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage, read> cellState: array<u32>;

        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
            let i = f32(input.instance);
            let state = f32(cellState[input.instance]);

            let cell = vec2f(i % grid.x, floor(i / grid.x));
            let cellOffset = cell / grid * 2.0;
            let gridPos = (input.position * state + 1.0) / grid - 1.0 + cellOffset;

            var output: VertexOutput;
            output.position = vec4f(gridPos, 0.0, 1.0);
            output.color = cell;

            return output;
        }

        @fragment
        fn fragmentMain(input: FragInput) -> @location(0) vec4f {
            let c = input.color / grid;
            return vec4f(c, 1.0 - c.x, 1.0);
        }
    `
});

const simulateShaderModule = device.createShaderModule({
    label: 'Game of Life Simulation Shader',
    code: `
        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage, read> cellStateIn: array<u32>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

        fn cellIndex(cell: vec2u) -> u32 {
            return (cell.y % u32(grid.y)) * u32(grid.x) 
                + (cell.x % u32(grid.x));
        }

        fn cellActive(x: u32, y: u32) -> u32 {
            return cellStateIn[cellIndex(vec2(x, y))];
        }

        @compute
        @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
            let activeNeighbors = 
                cellActive(cell.x + 1,  cell.y + 1) +
                cellActive(cell.x + 1,  cell.y) +
                cellActive(cell.x + 1,  cell.y - 1) +
                cellActive(cell.x,      cell.y - 1) +
                cellActive(cell.x - 1,  cell.y - 1) +
                cellActive(cell.x - 1,  cell.y) +
                cellActive(cell.x - 1,  cell.y + 1) +
                cellActive(cell.x,      cell.y + 1);

            let i = cellIndex(cell.xy);

            switch activeNeighbors {
                case 2u: {
                    cellStateOut[i] = cellStateIn[i];
                }
                case 3u: {
                    cellStateOut[i] = 1u;
                }
                default: {
                    cellStateOut[i] = 0u;
                }
            }
        }
    `
})

const bindGroupLayout = device.createBindGroupLayout({
    label: 'Cell Bind Group Layout',
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
            buffer: {
                type: 'uniform'
            }
        },
        {
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
            buffer: {
                type: 'read-only-storage'
            }
        },
        {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage'
            }
        }
    ]
});

const bindGroups = [
    device.createBindGroup({
        label: 'Cell Bind Group A',
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: uniformBuffer
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: cellStateBuffers[0]
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: cellStateBuffers[1]
                }
            }
        ]
    }),
    device.createBindGroup({
        label: 'Cell Bind Group B',
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: uniformBuffer
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: cellStateBuffers[1]
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: cellStateBuffers[0]
                }
            }
        ]
    })
];

const pipelineLayout = device.createPipelineLayout({
    label: 'Cell Pipeline Layout',
    bindGroupLayouts: [bindGroupLayout]
});

const cellPipeline = device.createRenderPipeline({
    label: 'Cell Pipeline',
    layout: pipelineLayout,
    vertex: {
        module: cellShaderModule,
        entryPoint: 'vertexMain',
        buffers: [
            {
                arrayStride: 2 * 4, // 2 floats per vertex, 4 bytes each
                attributes: [
                    {
                        format: 'float32x2',
                        offset: 0,
                        shaderLocation: 0
                    }
                ]
            }
        ]
    },
    fragment: {
        module: cellShaderModule,
        entryPoint: 'fragmentMain',
        targets: [
            {
                format: canvasFormat
            }
        ]
    }
});

const simulationPipeline = device.createComputePipeline({
    label: 'SImulation Pipeline',
    layout: pipelineLayout,
    compute: {
        module: simulateShaderModule,
        entryPoint: 'computeMain'
    }
})

let step = 0;

function updateGrid() {
    if (context === null)
        return;

    const commandEncoder = device.createCommandEncoder();

    const computePass = commandEncoder.beginComputePass();

    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);

    const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

    computePass.end();

    step++;

    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [
            {
                view: context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.4, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }
        ]
    });

    renderPass.setPipeline(cellPipeline);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setBindGroup(0, bindGroups[step % 2]);
    renderPass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE, 0, 0);

    renderPass.end();

    device.queue.submit([
        commandEncoder.finish()
    ]);
}

device.queue.writeBuffer(vertexBuffer, 0, vertices);
device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
device.queue.writeBuffer(cellStateBuffers[0], 0, cellStateArrayA);
device.queue.writeBuffer(cellStateBuffers[1], 0, cellStateArrayB);

setInterval(updateGrid, UPDATE_INTERVAL);