# AnalogBench 入门示例 A、B

完整命令与结果解释见 [Analog quickstart](../../../../docs/chips/ANALOG_QUICKSTART.md)。

- A：`rlc-rf-bandpass-100mhz`，无 PDK 的被动 RLC 设计，使用 ngspice。
- B：`sky130-ota-5t-gain40-pm60-noise50uv-pvt`，使用 ngspice 与镜像内的开放 SKY130 模型。

`fetch-sources.sh` 将固定版本原题下载到操作者指定的私有外部缓存。
本目录只保存接入代码。instruction、starter、solution、checker 和 PDK 均由上游提供。
Harbor 运行原题，维护 Agent、Trial、工作区隔离和评分生命周期。
