# 预训练快速添加 v3 交付记录

版本：[quick-add-v3.0.0-preview](https://github.com/DeadlinerStudio/DeadlinerCore/releases/tag/quick-add-v3.0.0-preview)。这是验收预览，质量门禁尚未完全通过。

| 评估 | 中文 | 英文 |
| --- | ---: | ---: |
| 总样例 | 1600 | 960 |
| 应可识别样例 | 1440 | 800 |
| 完整正确提案 | 1228 | 736 |
| 错误接受 | 0 | 0 |
| 完整正确覆盖 | 85.28% | 92.00% |

完整正确覆盖=类型、标题、日期/提醒、周期和备注全部匹配的已接受提案 / 应可识别样例。低置信度拒绝算未覆盖；没有剔除失败样例。中文达85%目标，但与英文差6.72个百分点，未达<=5点目标。两种语言的时间模板组合不同，分母1440/800；这不是所有自然输入的跨语言公平性证明。

g6有40组平行、未进入训练的任务标题；时间表达使用既有模板。冻结集在拟合前生成并记录SHA。它是作者构造的留出集，没有真人独立复核，不代表真实用户准确率。g4/g5曝光评估没有纳入训练，历史成绩与失败原因完整保存于docs/evidence/v3-pretrained.json。

## 训练与源码

中文改为UER tiny WWM的字符边界WordPiece底座，避免旧中文SentencePiece词合并任务/时间边界。中文依次6轮业务训练、3轮144组标题组合增强、2轮240组组合中抽取8000条增强；英文BERT tiny为4轮业务训练+2轮8000条组合增强。最后训练分区中文29703/英文27935条，开发集各900条。所有轮次seed42、lr5e-5、batch32，warm-start只恢复权重，训练配方与检查点SHA在发布证据与源码包。没有调用大模型生成用户提案。

只在训练分区增强；词表来自业务train及上游tokenizer。意图阈值0.85、margin0.25、所有字符槽位0.9、未知字符比例0.1，未为冻结集降低门限。所有模型原始revision、数据SHA、权重SHA见发布附件。训练入口pretrained_teacher，导出pretrained_export，实际ONNX/KMP评估benchmark，三格式/完整提案校验parity。

## 三端接线与一致性

共享KMP可选codec.wordPiece支持词表、大小写/去重音表、Unicode偏移、CLS/SEP、token BIO到字符映射；旧字符codec不变。所有有效评估输入逐条对照Hugging Face的完整ID和mask。超长不截断，未知/低置信/歧义拒绝，已有模型的语义拒绝不回退规则，AI手动触发。

Android: ONNX Runtime；iOS: Core ML CPU float32路径；鸿蒙: 系统MindSpore Lite。Core ML默认GPU半精度曾产生argmax差异，已保留失败证据并更换为CPU浮点。官方MindSporeLite2.7 converter生成.ms，ARM64 Linux官方运行库真实推理。华为官方转换指南由harmony-os-developer-knowledge复核，支持ONNX→.ms与Native OH_AI接口。

最终中文1639条、英文1001条（整个冻结集+已知回归集）三格式数值、有效token argmax及完整生产KMP提案完全一致；数值最大误差见附件。这不等价于iOS/鸿蒙真机验证。六个GitHub HTTPS文件均完整下载核对SHA/大小，服务端asset digest也匹配。

中文ONNX 12,710,586字节、英文17,520,314字节；按系统语言仅下载本平台对应权重，Sheet显示进度，成功后离线复用；权重不在安装包中。新版modelId使旧缓存不被采用。

## 回归与测试

已知回归：中文25/26、英文24/26完整正确，两组0错误接受；逐字前缀评估同时留档。前缀出现可接受提案或变化仅作为过程观测，不能宣称全部前缀无错误。

截图“后天晚上 8 点打电话”及4种空格/汉字数字变体均为打电话、2026-10-12T20:00（固定基准2026-10-10、Asia/Shanghai）。手写“每天下午 3 点交报告书”与无空格版本均为每日习惯、标题交报告书、提醒15:00；转钟与工作日回归见manual-probes.json。

训练工程28项测试通过；KMP/macOS快速添加58项和Android快速添加55项测试通过。扩大运行的全量Android 133项中，有5项既有通知测试因android.util.Log.i未mock而失败，未改动无关通知模块；保留quickadd-g5-kmp-tests.log。本轮不得宣称全量测试通过。

## 许可记录与待验证

中文HF模型卡未单独声明权重许可；其UER-py项目声明Apache-2.0。英文HF模型卡声明MIT。上游来源/许可按底座分别记录，未将所有预训练权重标成项目MIT；正式商用分发前仍需单独完成中文权重许可核实。

尚需独立人工金标、真实用户样例、三端真机P95/冷启动/下载UI与故障恢复验收。中文未识别的剩余表达应继续扩充独立训练/评测，不能据合成成绩推断真实通过率。

## 客户端产物

三端客户端构建均成功。最终产物均验证包含与发布文件逐字节一致的双语v3清单，APK/HAP ZIP完整、无嵌入权重，iOS资源与发布清单一致。

- Android arm64未签名APK：/Users/aritxonly/Codes/Android/Deadliner/app/release/app-arm64-v8a-v3-preview-small-unsigned.apk，37,352,123字节（37.35 MB），SHA 41b914447ad951a0aaf45554d0363feb53d26fac1875e6d7333adce71ed9453f。需要原发布密钥签名；未使用debug签名替代。ONNX JNI类名经R8保留，4个原生库均压缩。
- iOS Release未签名App：/Users/aritxonly/Codes/iOS/Deadliner/build/Build/Products/Release-iphoneos/Deadliner.app，CODE_SIGNING_ALLOWED=NO，需项目原有签名后安装。
- Harmony signed debug HAP：/Users/aritxonly/Codes/HarmonyOS/Deadliner/entry/build/default/outputs/default/entry-default-signed.hap，37059354字节，SHA 6fba1d05f88f4d7f1a1f01ac7736e568926af1d0b800aa01843785d83ca222c7。这是debug验收HAP，不是商店release。

共享库构建12m3s，Android release5m32s，Harmony debug5m51s；iOS包含再次构建XCFramework11m25s，随后Xcode Release BUILD SUCCEEDED。根构建脚本每次生成编译时间元数据，客户端会再次触发共享库构建，本轮未修改该无关流程。

精确产物记录见发布附件delivery-receipt.json；六个真实HTTPS下载校验与服务器digest记录均已保存。独立训练工程、Core规格校验、1000行新模块守卫、git diff --check通过。中文差距指标及真人/真机门禁继续留TODO，不能把打包成功视作模型最终验收成功。
