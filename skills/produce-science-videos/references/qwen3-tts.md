# Qwen3-TTS Base：部署、分段旁白与恢复

核对日期：2026-10-10。本文适用于 `scripts/run_qwen3_tts.py`。脚本本身为 MIT；Qwen 模型、官方代码与各依赖遵循各自许可证，不能由本仓库的 MIT 许可证替代。模型权重、真人参考声音、账号、私有稿件均不随仓库发布。

## 1. 适用范围与验证边界

- **已有实践背景**：本项目此前的旁白流程在 Windows、Python 3.12.14、Torch 2.10 / CUDA 12.8、Qwen3-TTS-12Hz-1.7B-Base、RTX 4080 Laptop GPU、BF16 上实际使用过。这是既有流程的经验，不是本次泛化脚本的逐平台认证。
- **本次已验证**：官方 API/配置核对；纯 Python 语法和导入；合成夹具的任务验证、签名、dry-run、恢复验证、无覆盖发布与输出路径防护。没有下载大模型，也没有用本脚本重新执行真实 GPU 或 CPU 推理。
- **云 CPU 经验**：本项目此前云 CPU 推理很慢，不建议直接拿长篇旁白试跑。没有可复现的正式速度基准，不提供臆测的倍速、显存下限或交付时间。
- **未验证**：其他 GPU/驱动、macOS/MPS、ROCm、FlashAttention 构建、网络盘与各云服务镜像。本脚本只提供 `cpu` 和 `cuda:N` 路径，不宣称涵盖所有后端。

先用一个短段落完成真实推理和听审，再扩展到全片。文件可读、SHA 正确只代表工程完整性，不代表句子读全、术语读对或声音自然。

## 2. 模型与官方 API

选择 **Base** 模型做参考音频克隆；不要把 CustomVoice/VoiceDesign 的 `speaker` 或 `instruct` 参数直接套到 Base。本脚本默认使用带参考转写的 ICL 模式：

```python
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(local_model_dir, **load_options)
prompt = model.create_voice_clone_prompt(
    ref_audio=(mono_waveform, sample_rate),
    ref_text=reference_transcript,
    x_vector_only_mode=False,
)
wavs, sample_rate = model.generate_voice_clone(
    text=paragraph,
    language="Chinese",
    voice_clone_prompt=prompt,
    non_streaming_mode=True,
    max_new_tokens=2048,
)
```

模型每次进程加载一次，参考 prompt 创建一次，每段串行调用。官方也支持只用 speaker embedding 的 `x_vector_only_mode=True`，但它省去参考文本/语音码条件，不能当作同一模式的无损替代，本 CLI 不启用。API 返回波形列表与实际采样率；脚本不硬编码采样率。官方完整示例见 [Base 示例](https://github.com/QwenLM/Qwen3-TTS/blob/main/examples/test_model_12hz_base.py) 与 [API 实现](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py)。

只使用有权使用、获得本人许可的参考声音，明确用途。不得用声音克隆冒充他人或制造其发表过某段话的假象。录音应为单声道、无音乐、无明显混响和串音；参考转写必须忠实于录音，不是待合成的新稿。时长与质量应先用短样试听确定，不把营销中的“几秒克隆”当作质量保证。

## 3. 环境准备：在独立环境中完成

1. 优先保留现有、已经成功的环境。新环境建议 Python 3.12；官方 [安装说明](https://github.com/QwenLM/Qwen3-TTS#environment-setup) 也采用该版本。
2. 按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 选择操作系统和 CUDA/CPU 构建，并安装匹配的 torch/torchaudio。不要为一个旁白任务升级整个系统的 CUDA、驱动或其他项目依赖。
3. 本次核对到官方 [pyproject.toml](https://github.com/QwenLM/Qwen3-TTS/blob/main/pyproject.toml) 声明 `qwen-tts` 版本 `0.1.1`、`transformers==4.57.3`、`accelerate==1.12.0`。可在新的隔离环境中从这个可解释的起点安装：

```text
python -m venv .venv
# Linux/macOS 激活：source .venv/bin/activate
# Windows 可直接使用 .venv\Scripts\python.exe，不需改变 PowerShell 执行策略。
python -m pip install "qwen-tts==0.1.1"
python -m pip check
python -m pip freeze > environment-lock.txt
```

激活环境后才使用上面的 `python`；Windows 不激活时，把它替换为 `.venv\Scripts\python.exe`。先完成第 2 步的 PyTorch 安装，再装 qwen-tts。这里是部署示例，不是已在所有平台测试的完整锁文件。安装后保存包版本、GPU 名称、驱动和模型 revision；已有可用环境以其实际锁文件为准。`environment-lock.txt` 应保存在自己的工作目录，提交公开仓库前检查是否含私有路径或索引地址。

FlashAttention 2 是官方建议的可选 GPU 优化，需要兼容硬件与 FP16/BF16；本脚本不自动安装它。初次运行不要为了它覆盖可用环境。CPU 用 FP32；GPU 显式选择 BF16 或 FP16，不静默回退到 CPU。

### 获取完整的本地模型快照

在运行旁白之前，单独从 [官方 Base 模型页](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) 选择固定 commit 并下载完整快照。官方文件树含 `speech_tokenizer/`，不要只取顶层权重。已有完整本地模型无需重新下载。

已安装兼容的 Hugging Face CLI 时，可按 [官方 CLI 文档](https://huggingface.co/docs/huggingface_hub/guides/cli) 使用以下模板；先把 `MODEL_COMMIT` 替换为模型历史中的真实完整 commit：

```text
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --revision MODEL_COMMIT --local-dir ./models/qwen3-tts-base
```

下载是单独的部署步骤，会使用网络与磁盘。runner 不接受远程 model ID，不安装依赖，不下载模型；真实推理前设置 Hugging Face/Transformers 离线变量并传入 `local_files_only=True`。缺文件时停止，不悄悄切换模型或远程服务。离线变量不是操作系统网络沙箱，严格断网要求应由部署环境另外保证。

`--model-revision` 是必填的真实快照/发布标识，不是下载指令或权重校验机制。脚本只哈希少量配置与 tokenizer 文件，不扫描或哈希数 GB 的权重，也没有固定权重摘要门禁。运行期间保持模型目录只读/不变；原地更换权重却沿用 revision 不在该输入签名的检测保证内。

## 4. 本地、云 GPU、云 CPU 的选法

| 环境 | 起步参数 | 注意事项 |
| --- | --- | --- |
| 已有 Windows NVIDIA GPU | `--device cuda:0 --dtype bfloat16 --attention default` | 先沿用成功环境并试一个短段。BF16 不支持会报错，可另选 FP16；不自动替换。 |
| 自管 Linux / 云 GPU | 同上，按硬件选择 BF16/FP16 | 配置持久化模型盘和输出盘；检查 CUDA 可见性；在新镜像重新做短样验收。云租用、费用与录音上传须另获授权。 |
| 本地 / 云 CPU | `--device cpu --dtype float32 --attention default` | 适合兼容性短测；本项目经验不建议长篇。先测实际耗时，确认预算后再扩大。 |

`--attention default` 不向模型加载器强制设置 attention 后端，由所固定的官方包/配置决定。`eager`、`sdpa`、`flash_attention_2` 是显式选项，不保证每个内部组件或每个安装版本都接受。选择与实际工作环境匹配的组合，并记录它。脚本不会自动试一串后端，也不会替换内部函数。

### tokenizer / eager 兼容问题

既有部署曾遇到 speech tokenizer 的配置与 eager attention 组合问题；外层模型设为 eager，不一定同时改变内部 tokenizer 的加载行为。这是版本相关的经验，不是所有安装都会出现的固定故障。

优先固定已经成功的 qwen-tts、Transformers、Torch 与模型快照。新环境遇到同类错误时保存完整异常和版本，核对 [官方模型加载实现](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py) 及上游修复，在独立环境验证一个短样。不要自动猴补丁、批量改 site-packages、改整个系统或宣称设置一个 `--attention eager` 就解决了全部组件问题。本通用 runner 没有复刻历史环境的私有修补。

## 5. Jobs 与运行

保存为 UTF-8 JSON 数组，字段必须恰好为 `id`、`text`、`language`、`seed`：

```json
[
  {"id": "demo_01", "text": "这是一个计时演示。", "language": "Chinese", "seed": 101},
  {"id": "demo_02", "text": "第二段保持同一条时钟。", "language": "Chinese", "seed": 102}
]
```

ID 为 1–80 个 ASCII 字母、数字、下划线或短横线，首字符须是字母或数字；拒绝 Windows 保留名、路径、重复及仅大小写不同的 ID。seed 是 `0` 至 `2**32-1` 的整数，不能为布尔值。每段使用固定 seed 便于对照，但跨 GPU、软件版本和算子不承诺逐样本相同。

支持的语言名为 Auto、Chinese、English、Japanese、Korean、German、French、Russian、Portuguese、Spanish、Italian；已知语言应显式指定。用自然语义边界拆段，在上游把公式、人名和数字处理成适合朗读的文本。不要把整部长片塞进一个 job；`max_new_tokens` 是上限，输出可读并不能排除达到上限导致的截断。

以下均从 skill 目录运行。路径只是虚构模板，需替换为自己的文件；`voice-reference.txt` 内容为对应录音的准确转写。

```text
python scripts/run_qwen3_tts.py --jobs ./jobs.json --model-dir ./models/qwen3-tts-base --model-revision MODEL_COMMIT --ref-audio ./voice-reference.wav --ref-text-file ./voice-reference.txt --output-dir ./audio-run-01 --device cuda:0 --dtype bfloat16 --dry-run
```

确认计划后移除 `--dry-run` 执行。CPU 使用 `--device cpu --dtype float32`。续跑保留完全相同的路径、输入和参数，添加 `--resume`；可以先 `--resume --dry-run`。不要同时开多个 runner 抢占同一块 GPU。

### dry-run 能证明什么

- 使用标准库检查 JSON、ID、路径、Base 配置、签名和已有输出；不导入 torch/numpy/soundfile，不创建输出目录、不联网、不加载权重。
- 参考音频只检查存在并计算 SHA，**不解码**；不证明参考录音可用、权重完整、GPU 足够、包安装兼容或实际推理成功。
- 没有 torch 的机器仍可 dry-run；若包安装状态或版本后来改变，签名也改变。请在最终运行环境重新 dry-run。

## 6. 恢复、输出与失败处理

每段输出 `ID.wav` 与 `ID.json`。元数据记录输入签名、文本/seed、模型路径与 revision、配置摘要、参考音频/文本摘要、参数、运行库版本、时长和 WAV SHA。它们属于自己的生产资料，默认应保存在私有工作目录，不能不经检查提交公开仓库。

- 只有输入签名一致、完成记录正确、单声道 PCM16 WAV 可完整读取、非空/非全零、帧数/采样率/字节数等全部一致且完整文件 SHA 相同，`--resume` 才跳过该段。
- 默认 attention 和采样值来自模型配置及所安装官方包，相关配置摘要与包版本纳入签名。runner 自身变动也会改变签名；更新脚本后使用新的输出目录。
- 缺一半文件、残留 `.partial`、损坏 WAV/JSON、不同输入或意外链接都会报错。脚本不会自动覆盖、删除旧成品或用错误文件“凑齐进度”。
- 每段先写同目录唯一临时文件，flush/fsync 后通过 hard-link 原子地发布到不存在的文件名；目标存在则失败。成功后只移除本次临时名字。WAV 与 JSON 是两个独立提交；中途停止产生的孤立文件在续跑时明确拒绝。
- 该无覆盖机制要求本地文件系统支持 hard links，例如 NTFS/ext4。FAT、某些云挂载或网络盘可能不支持，错误时停下并改用私有本地输出目录；不要自动降级到覆盖写。它不是分布式事务，也不保证断电后的目录项持久性。
- 输出路径拒绝 symlink/junction，段 ID 无法逃出输出目录；请使用仅自己可写的目录。它不防御恶意进程在检查后替换目录，也不允许将输出写入模型目录。
- `.qwen3-tts.lock` 防止同一输出目录被本 runner 并行使用。进程被强制终止可能留下锁；先确认原进程已结束，再人工保存/审查残留并清理锁。不要看到锁就自动删除。
- 修复重跑：保留失败文件与日志，使用新的输出目录或人工移动整组冲突文件到审查目录。没有 `--force`，没有静默“自愈”。

音频输出为 PCM16 单声道；转换前检查有限值、非静音和峰值范围，超范围会拒绝而不是静默裁切。脚本不去除头尾、不响度归一化、不变速、不变调。量化属于保存格式转换；如需浮点母版，应另行明确扩展格式与验证器，不能仅改后缀。

## 7. 最小配音交接包

用于把已冻结旁白交给另一台已具备推理环境的机器。沿用可用部署，不要求先重装环境、下载同一模型或增加一套新的重型校验门禁。

- `jobs.json`：固定段 ID、准确朗读文本、language 与 seed；附冻结的完整旁白文本及其摘要，明确修改稿件必须重新确认。
- 当前 `run_qwen3_tts.py`，实际可用的依赖版本记录、模型名称与固定 revision；执行说明填写接收机器自己的本地模型目录，不把权重装进交接包。
- 参考音频与准确转写单独走获准的私有渠道；交接前明确声音使用许可、接收者/云服务、用途与保留范围。没有许可就不上传声音。公开示例包不含真人参考音频。
- 一条短执行说明，例如：“使用现有环境和指定本地 Base 模型，先 dry-run；通过后只合成已冻结 jobs，模型和 prompt 各加载一次，逐段串行。遇到冲突保留原文件并报告。按段或按集返还已完成 WAV、对应 JSON 与 manifest；列出失败/待生成段和实际设备，听审状态保持未审。”
- 返还包仅放本次应交付且已验证的 WAV/JSON，加文件 manifest（相对路径、字节数、SHA-256）和短运行报告。运行报告写实际 CPU/GPU、dtype、attention、库版本、完成/未完成 ID；不能用“指定 cuda:0”代替实际 GPU 名称。当前 runner 记录请求的 device 与版本，实际 GPU 名称/驱动须由执行者另行核实补入报告。

按段或按集建立返还目录后，可用已有清单工具生成并验回；解包到新目录再次验证：

```text
python scripts/file_manifest.py create ./audio-return --output ./audio-return/manifest.json
python scripts/file_manifest.py verify ./audio-return ./audio-return/manifest.json
```

不要把模型、参考录音、令牌、环境缓存或失败 `.partial` 塞进成品包。部分完成可以先交付，但必须列明应有/实有段数，不能把部分包标为全部完成。manifest 证明文件完整性，报告中的“未听审”仍需保留，直到实际听审完成。

## 8. 工程完成与主观听审分开

`inference_complete` 只表示本段推理返回并通过文件验收。每段初始记录始终是 `semantic_listening_performed: false`、`listening_status: pending`。全部推理完成不等于视频已可发布。

逐段人工听审并在独立 QA 记录中写明：段 ID、对应 WAV SHA、审听人/日期、通过或问题点、是否需要重新生成。核对漏词/复读、句末截断、人名与科学术语、数字单位、语义重音、爆音与参考声音一致性。ASR/文本比对可以辅助，不能代替自然度和科学含义听审。重生成改变 WAV SHA 后，原听审结论失效。

## 9. 可测试接口

可以使用 `importlib.util.spec_from_file_location` 加载脚本，不会导入推理依赖。可单测函数包括 `validate_jobs`、`canonical_hash`、`input_signature`、`wav_info`、`check_existing`、`safe_output_root` 和 `publish_no_clobber`。

本次使用临时合成 WAV/JSON 夹具通过：合法任务、越界/保留 ID、大小写重复、错误 seed/语言、签名稳定性与输入变化、dry-run 无输出写入/无推理导入、正确恢复、签名不匹配、截断 WAV、孤立输出、目标存在时不覆盖、symlink 路径拒绝。测试不使用任何真实声音或长视频；这些测试不能替代真实 Qwen 推理与人工听审。
