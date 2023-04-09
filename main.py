from riscv import RiscVProgram

def test():
    t1 = RiscVProgram("""
Test1:
    add a5, s0, a5
Test2:
    add a5, s0, a5
Test3:
    add a5, s0, a5
""")
    
    # add a5, s0, a5
    # add : 0110011
    # a5 : 15 -> 01111
    # s0 : 8 -> 01000
    # 0000000(funct7) 01111(a5) 01000(s0) 000(funct3) 01111(a5) 0110011(add)
    
    t1.parse()
    t1.show_hex()
    t1.inspect_all()

def main():
    p = RiscVProgram("""
Assignment1:
    addi sp, sp, -32
    sw ra, 28(sp)
    sw s0, 24(sp)
    sw a0, 12(sp)
    lw a4, 12(sp)
    addi a5, zero, 1
    beq a4, a5, L1
    lw a4, 12(sp)
    addi a5, zero, 2
    bne a4, a5, L2
L1:
    addi a5, zero, 1
    jal zero, L3
L2:
    lw a5, 12(sp)
    addi a5, a5, -1
    addi a0, a5, 0
    jal ra, Assignment1
    addi s0, a0, 0
    lw a5, 12(sp)
    addi a5, a5, -2
    addi a0, a5, 0
    jal ra, Assignment1
    addi a5, a0, 0
    add a5, s0, a5
L3:
    addi a0, a5, 0
    lw ra, 28(sp)
    lw s0, 24(sp)
    addi sp, sp, 32
    jalr zero, 0(ra)
""")

    p.parse()
    p.inspect_all()
    # p.check()
    p.show_hex()


main()