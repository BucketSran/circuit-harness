module tb;
  reg a, b;
  wire y;
  adder dut(a, b, y);
  integer i;
  initial begin
    $dumpfile("wave.vcd");
    $dumpvars(0, tb);
    for (i = 0; i < 4; i = i + 1) begin
      {a, b} = i;
      #1;
      if (y !== (a ^ b)) $fatal(1, "wrong truth table");
    end
    $display("CHIPS_TEST_PASS");
    $finish;
  end
endmodule
