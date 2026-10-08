# D0 — differential gain measurement request

This is a synthetic single-pole behavioral amplifier for testing the authoring workflow.
Please build a testbench to measure its differential voltage gain at 20 Hz.

Operating conditions: supply 1200 mV, both input DC common-mode levels 600 mV,
temperature 25 degC, and a 2 megaohm load from output to ground.
The required voltage-gain magnitude is at least 50 dB.

The ordered DUT interface is shown only in d0-interface.png. Preserve that order
when instantiating the model. Node `0` means ground. The image is an interface
contract, not a transistor schematic or a proposed AC excitation.
