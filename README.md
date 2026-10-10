# Popular Science Video Skills · 科普视频制作工作流

由 **Dot 生成与整理**，把一次真实科普视频制作中的研究、叙事、配音、画面、配乐、字幕与恢复交付经验，整理成可复用的技能和轻量 Python 工具。

它帮助你把一项研究讲成有证据、有动作、有边界的故事。不是一键视频生成器，也不附带模型、素材或某个项目的成片。

## 先知道这三件事

1. **工作流仅在 Dot 中实践过。** 本仓库公开版脚本另做了合成数据测试；不表示完整工作流已在其他助手、编辑器或操作系统中端到端验证。
2. **先做一集，再扩批量。** 先跑通研究、声音、整片时序、视觉与听审；漂亮的样段或很多镜头，不能替代完整成片。
3. **素材和授权由使用者准备。** 本仓库不含真实视频、音频、照片、模型权重、声音参考、声纹缓存或私有工程。MIT 只覆盖本仓库的原创文字与代码。

## 快速上手

### 让助手使用技能

把 `skills/produce-science-videos/` 放进你的助手支持的技能目录，或让助手读取其中的 `SKILL.md`。不同宿主的安装位置、工具和权限可能不同，请按该宿主文档适配；这里没有实测通用安装器。

一个请求示例：

> 请用 produce-science-videos 工作流，制作一个面向普通观众的 3 分钟中文科普视频。先核对原始研究并给我主张表、旁白与分镜；确认后再制作。画幅、声音、素材来源和交付位置先按项目配置核对。

复制 [项目配置模板](skills/produce-science-videos/assets/project-config.template.json)，只填本次任务需要的字段。长度、语言、16:9/其他画幅、配音方式、字幕密度、音乐强弱和终审方式都是配置项，不是全局固定偏好。

### 不下载模型，先试轻量工具

基础工具只用 Python 标准库。媒体检查额外需要可执行的 FFmpeg/ffprobe。下列命令从仓库根目录运行：

```bash
python -m unittest discover -s tests -v
python skills/produce-science-videos/scripts/timeline_plan.py examples/minimal/timing.json
python skills/produce-science-videos/scripts/captions.py examples/minimal/captions.json --output /tmp/demo.srt
python skills/produce-science-videos/scripts/file_manifest.py create examples/minimal --output /tmp/demo-manifest.json
python skills/produce-science-videos/scripts/file_manifest.py verify examples/minimal /tmp/demo-manifest.json
```

Windows 可把 `/tmp/demo.srt` 等替换成自己的临时输出路径。仓库只有 `examples/minimal/` 这一个合成小案例；不依赖外部素材。生成测试文件留在临时目录，不需要提交。

### 使用 Qwen3-TTS 配音

先读 [Qwen3-TTS 本地、云端与交接指南](skills/produce-science-videos/references/qwen3-tts.md)。其中包含环境准备、Base 声音复用、一次载入、串行补缺、短执行指令与回传清单。

- 已有可用部署：优先沿用，别为了工作流再装一套。
- 新部署：单独环境，按官方资料选择 PyTorch/CUDA；先短段验证。
- 云 GPU：同样需要确认声音参考允许上传到该提供商。
- 云 CPU：在实际尝试中很慢，不建议直接排长篇任务；没有可泛化的速度基准。
- 泛化 runner 的 GPU/CPU 真实推理在本次公开版验证中**没有重跑**，请勿把单元测试当作模型兼容承诺。

## 你会得到什么

| 内容 | 入口 |
|---|---|
| 按需加载的主技能 | [SKILL.md](skills/produce-science-videos/SKILL.md) |
| 新研究、奖项与历史资料核验 | [研究与叙事](skills/produce-science-videos/references/research-story.md) |
| 画面、音乐、字幕、封面 | [声音与视觉](skills/produce-science-videos/references/audio-visual.md) |
| Qwen3-TTS 部署和配音交接 | [Qwen3-TTS](skills/produce-science-videos/references/qwen3-tts.md) |
| 样本时钟、缓存与中断恢复 | [生产与存储](skills/produce-science-videos/references/production-storage.md) |
| 实际视听、技术检查与交付 | [验收与交付](skills/produce-science-videos/references/quality-delivery.md) |
| 模型分工、测试边界 | [验证记录](docs/VALIDATION.md) |
| 第三方许可和公开前检查 | [许可边界](docs/RIGHTS_AND_PRIVACY.md) |

工具各自独立，不会自动发布、上传、删文件、替你同意许可或安装模型。用 `--help` 查看参数。

## 实际模型与分工

这是来源项目的实际分工记录，不要求复用者选择相同模型：

- **GPT-6 Astra（xhigh）**：主导科普视频任务，进行研究组织、故事/分镜、生产协调与独立审阅。
- **GPT-6 Luna**：本地配音任务的调度助手；不是生成声音的 TTS 模型。
- **Qwen3-TTS-12Hz-1.7B-Base**：实际旁白合成模型；Windows、Python 3.12.14、PyTorch 2.10 / CUDA 12.8、RTX 4080 Laptop、BF16 的已有部署完成过生产。
- **imagegen**：实际调用的原生图像生成/编辑能力。记录未暴露底层图像模型名称，因此不猜型号。
- **Pillow / NumPy / FFmpeg**：程序化画面、动画与编码，不是生成式视觉模型。
- **Vosk 0.3.45 + vosk-model-small-cn-0.22**：辅助发现中文语音边界和识别风险，不能取代原文、科学术语核对或人工听审。

更细的验证范围见 [VALIDATION.md](docs/VALIDATION.md)。这里不公开原项目的私人输入和日志。

## English summary

Evidence-led science-video production guidance, generated and organized by Dot. The production workflow has only been exercised in Dot. This public version includes lightweight tested utilities and one synthetic example; integrations with other hosts and real inference using the generalized TTS runner remain unverified. No media, model weights, voice references, or private project data are distributed.

## License

[MIT](LICENSE) applies to the original workflow text and code in this repository. Third-party models, software, music, fonts, photographs, portraits and reference voices retain their own terms and permissions. A model or asset being downloadable does not establish the right to use or redistribute it.
