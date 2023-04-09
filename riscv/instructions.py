from abc import ABCMeta, abstractmethod
from enum import IntEnum, StrEnum, auto
from pprint import pprint
import re
from typing import ClassVar, Final, Optional, Type, cast

from riscv.utils import bin_str


__all__ = (
    "RiscVTokenType",
    "RiscVToken",
    "Label",
    "RiscVOpCodes",
    "RiscVFunct3",
    "RiscVFunct7",
    "OPCode",
    "RiscVRegisters",
    "Register",
    "RiscVInstructionType",
    "RiscVInstruction",
    "RegisterInstruction",
    "parse_instruction"
)


class RiscVTokenType(StrEnum):
    OPCODE = auto()
    FUNCT3 = auto()
    FUNCT7 = auto()
    RD = auto()
    RS1 = auto()
    RS2 = auto()
    IMM = auto()
    LABEL = auto()
    SPECIAL = auto()


class RiscVToken(metaclass=ABCMeta):
    """RISC-V token"""
    type: RiscVTokenType
    
    def __init__(self, type: RiscVTokenType):
        self.type = type
    
    @abstractmethod
    def to_binary(self) -> str:
        """Generate binary code of this token. Used in RiscVInstruction.to_binary()"""
        ...
    
    def __str__(self) -> str:
        return f"RiscV.Token.{self.type}"

class Label(RiscVToken):
    """
    Special token to indicate label.

    Args:
        name (str) : name of the label
        value (int) : pc of the label
    """
    name: str
    pc: int
    
    def __init__(self, name: str, pc: int):
        super().__init__(type=RiscVTokenType.LABEL)
        self.name = name
        self.pc = pc
    
    def to_binary(self) -> str:
        return bin_str(self.pc, 12)
    
    def __str__(self) -> str:
        return super().__str__() + f"({self.name}:{self.pc})"
    
    __repr__ = __str__
    
    def __int__(self) -> int:
        return self.pc

# Just for now...

class RiscVOpCodes(IntEnum):
    # U types
    LUI = int("0110111", base=2)
    AUIPC = int("0010111", base=2)
    # J types
    JAL = int("1101111", base=2)
    # I types
    JALR = int("1100111", base=2)
    # B types
    BEQ = int("1100011", base=2)
    BNE = int("1100011", base=2)
    BLT = int("1100011", base=2)
    BGE = int("1100011", base=2)
    BLTU = int("1100011", base=2)
    BGEU = int("1100011", base=2)
    
    LB = int("0000011", base=2)
    LH = int("0000011", base=2)
    LW = int("0000011", base=2)
    LBU = int("0000011", base=2)
    LHU = int("0000011", base=2)
    
    SB = int("0100011", base=2)
    SH = int("0100011", base=2)
    SW = int("0100011", base=2)
    
    ADDI = int("0010011", base=2)
    SLTI = int("0010011", base=2)
    SLTIU = int("0010011", base=2)
    XORI = int("0010011", base=2)
    ORI = int("0010011", base=2)
    ANDI = int("0010011", base=2)
    SLLI = int("0010011", base=2)
    SRLI = int("0010011", base=2)
    SRAI = int("0010011", base=2)
    
    ADD = int("0110011", base=2)
    SUB = int("0110011", base=2)
    SLL = int("0110011", base=2)
    SLT = int("0110011", base=2)
    SLTU = int("0110011", base=2)
    XOR = int("0110011", base=2)
    SRL = int("0110011", base=2)
    SRA = int("0110011", base=2)
    OR = int("0110011", base=2)
    AND = int("0110011", base=2)
    
    FENCE = int("0001111", base=2)
    
    ECALL = int("1110011", base=2)
    EBREAK = int("1110011", base=2)

class RiscVFunct3(IntEnum):
    """Fuct3 part of each opcode."""
    # opcode : 1100111
    JALR = int("000", base=2)
    BEQ = int("000", base=2)
    BNE = int("001", base=2)
    BLT = int("100", base=2)
    BGE = int("101", base=2)
    BLTU = int("110", base=2)
    BGEU = int("111", base=2)
    
    # opcode : 0000011
    LB = int("000", base=2)
    LH = int("001", base=2)
    LW = int("010", base=2)
    LBU = int("100", base=2)
    LHU = int("101", base=2)
    
    # opcode : 0100011
    SB = int("000", base=2)
    SH = int("001", base=2)
    SW = int("010", base=2)
    
    # opcode : 0010011
    ADDI = int("000", base=2)
    SLTI = int("010", base=2)
    SLTIU = int("011", base=2)
    XORI = int("100", base=2)
    ORI = int("110", base=2)
    ANDI = int("111", base=2)
    
    SLLI = int("001", base=2)
    SRLI = int("101", base=2)
    SRAI = int("101", base=2)
    
    # opcode : 0110011
    ADD = int("000", base=2)
    SUB = int("000", base=2)
    SLL = int("001", base=2)
    SLT = int("010", base=2)
    SLTU = int("011", base=2)
    XOR = int("100", base=2)
    SRL = int("101", base=2)
    SRA = int("101", base=2)
    OR = int("110", base=2)
    AND = int("111", base=2)
    
    # opcode : 1110011
    FENCE = int("000", base=2)
    
    # opcode : 1110011
    ECALL = int("000", base=2)
    EBREAK = int("000", base=2)

class RiscVFunct7(IntEnum):
    """Func7 part of each opcode."""
    # opcode : 0010011
    SLLI = int("0000000", base=2)
    SRLI = int("0000000", base=2)
    SRAI = int("0000001", base=2)
    
    # opcode : 0110011
    ADD = int("0000000", base=2)
    SUB = int("0100000", base=2)
    SLL = int("0000000", base=2)
    SLT = int("0000000", base=2)
    SLTU = int("0000000", base=2)
    XOR = int("0000000", base=2)
    SRL = int("0000000", base=2)
    SRA = int("0100000", base=2)
    OR = int("0000000", base=2)
    AND = int("0000000", base=2)


class OPCode(RiscVToken):
    cmd: str
    # values to generate binary
    opcode: RiscVOpCodes
    funct3: Optional[RiscVFunct3]
    funct7: Optional[RiscVFunct7]
    
    def __init__(self, cmd: str):
        super().__init__(type=RiscVTokenType.OPCODE)
        self.cmd = cmd
        
        cmd_up = cmd.upper()
        self.opcode = RiscVOpCodes[cmd_up]
        self.funct3 = RiscVFunct3[cmd_up] if cmd_up in RiscVFunct3.__members__ else None
        self.funct7 = RiscVFunct7[cmd_up] if cmd_up in RiscVFunct7.__members__ else None
    
    def __str__(self) -> str:
        return super().__str__() + f"({self.cmd})"
    
    def to_binary(self) -> str:
        b = bin_str(self.opcode.value, 7)
        
        if self.funct3 is not None:
            f3: str = bin_str(self.funct3.value, 3)
            b += "," + f3
        
        if self.funct7 is not None:
            f7: str = bin_str(self.funct7.value, 7)
            b += "," + f7
        
        return b
    
    def inspect(self):
        print(f"opcode : {self.opcode.name} ({bin_str(self.opcode.value, 7)})")
        if self.funct3 is not None:
            print(f"funct3 : {self.funct3.name} ({bin_str(self.funct3.value, 3)})")
        if self.funct7 is not None:
            print(f"funct7 : {self.funct7.name} ({bin_str(self.funct7.value, 7)})")


class RiscVRegisters(IntEnum):
    """RiscV register names"""
    
    zero = x0 = 0
    ra = x1 = 1
    sp = x2 = 2
    gp = x3 = 3
    tp = x4 = 4
    t0 = x5 = 5
    t1 = x6 = 6
    t2 = x7 = 7
    s0 = fp = x8 = 8
    s1 = x9 = 9
    a0 = x10 = 10
    a1 = x11 = 11
    a2 = x12 = 12
    a3 = x13 = 13
    a4 = x14 = 14
    a5 = x15 = 15
    a6 = x16 = 16
    a7 = x17 = 17
    s2 = x18 = 18
    s3 = x19 = 19
    s4 = x20 = 20
    s5 = x21 = 21
    s6 = x22 = 22
    s7 = x23 = 23
    s8 = x24 = 24
    s9 = x25 = 25
    s10 = x26 = 26
    s11 = x27 = 27
    t3 = x28 = 28
    t4 = x29 = 29
    t5 = x30 = 30
    t6 = x31 = 31
    shamt = -1


class Register(RiscVToken):
    def __init__(self, type: RiscVTokenType, address: RiscVRegisters):
        super().__init__(type)
        self.address = address
    
    def to_binary(self) -> str:
        return bin_str(self.address.value, 5)
    
    def inspect(self):
        print(f"{self.type.name.lower()} : {self.address.name} ({self.to_binary()})")
    
    def __str__(self) -> str:
        return self.type.name


SHAMT: Final[Register] = Register(type=RiscVTokenType.SPECIAL, address=RiscVRegisters.shamt)


class RiscVInstructionType(StrEnum):
    R = auto()
    I = auto()
    S = auto()
    B = SB = auto()
    U = auto()
    J = auto()


InstructionTypeDictionary: dict[str, RiscVInstructionType] = {
    "lui": RiscVInstructionType.U,
    "auipc": RiscVInstructionType.U,
    "jal": RiscVInstructionType.J,
    "jalr": RiscVInstructionType.I,
    "beq": RiscVInstructionType.B,
    "bne": RiscVInstructionType.B,
    "blt": RiscVInstructionType.B,
    "bge": RiscVInstructionType.B,
    "bltu": RiscVInstructionType.B,
    "bgeu": RiscVInstructionType.B,
    "lb": RiscVInstructionType.I,
    "lh": RiscVInstructionType.I,
    "lw": RiscVInstructionType.I,
    "lbu": RiscVInstructionType.I,
    "lhu": RiscVInstructionType.I,
    "sb": RiscVInstructionType.S,
    "sh": RiscVInstructionType.S,
    "sw": RiscVInstructionType.S,
    "addi": RiscVInstructionType.I,
    "slti": RiscVInstructionType.I,
    "sltiu": RiscVInstructionType.I,
    "xori": RiscVInstructionType.I,
    "ori": RiscVInstructionType.I,
    "andi": RiscVInstructionType.I,
    "slli": RiscVInstructionType.R,
    "srli": RiscVInstructionType.R,
    "srai": RiscVInstructionType.R,
    "add": RiscVInstructionType.R,
    "sub": RiscVInstructionType.R,
    "sll": RiscVInstructionType.R,
    "slt": RiscVInstructionType.R,
    "sltu": RiscVInstructionType.R,
    "xor": RiscVInstructionType.R,
    "srl": RiscVInstructionType.R,
    "sra": RiscVInstructionType.R,
    "or": RiscVInstructionType.R,
    "and": RiscVInstructionType.R,
    "fence": RiscVInstructionType.I,        # imm part replaced with [ fm / pred / succ ]
    "ecall": RiscVInstructionType.I,
    "ebreak": RiscVInstructionType.I,
}

class RiscVInstruction(metaclass=ABCMeta):
    """RISC-V instruction"""
    type: RiscVInstructionType
    required_args: ClassVar[tuple[str, ...]] = ("cmd", )
    
    def __init__(self, type: RiscVInstructionType) -> None:
        self.type = type
    
    @abstractmethod
    def to_binary(self) -> str:
        """
        Encode this instruction into binary presentation.

        Returns:
            str: string-formed binary presentation of this instruction.
        """
        return sum((getattr(self, arg) for arg in reversed(self.required_args)), "")
            
    
    def to_hex(self) -> str:
        """
        Encode this instruction into hexademical presentation.

        Returns:
            str: string-formed binary presentation of this instruction.
        """
        return hex(int(self.to_binary(), base=2))
    
    def inspect(self):
        print(f"{self} > ", end="")
        for arg in self.required_args:
            print(f"{arg}={getattr(self, arg)}", end=",")
        print()
    
    def __str__(self) -> str:
        return f"RiscV.Instruction.{self.type.name}"
    
    __repr__ = __str__

class RegisterInstruction(RiscVInstruction):
    required_args: tuple[str, ...] = ("cmd", "rd", "rs1", "rs2")
    cmd: OPCode     # funct3, funct7 in here
    rd: Register
    rs1: Register
    rs2: Register
    
    def __init__(self, cmd: str, rd: str, rs1: str, rs2: str):
        super().__init__(type=RiscVInstructionType.R)
        self.cmd = OPCode(cmd)
        self.rd = Register(RiscVTokenType.RD, RiscVRegisters[rd])
        self.rs1 = Register(RiscVTokenType.RS1, RiscVRegisters[rs1])
        self.rs2 = Register(RiscVTokenType.RS2, RiscVRegisters[rs2])
    
    def to_binary(self) -> str:
        opcode, funct3, funct7 = self.cmd.to_binary().split(",")
        rd: str = self.rd.to_binary()
        rs1: str = self.rs1.to_binary()
        rs2: str = self.rs2.to_binary()
        return funct7 + rs2 + rs1 + funct3 + rd + opcode
    
    def inspect(self):
        super().inspect()
        
        self.cmd.inspect()
        opcode, funct3, funct7 = self.cmd.to_binary().split(",")
        
        self.rd.inspect()
        rd: str = self.rd.to_binary()
        
        self.rs1.inspect()
        rs1: str = self.rs1.to_binary()
        
        self.rs2.inspect()
        rs2: str = self.rs2.to_binary()
        
        print(f"total -> {funct7 + rs2 + rs1 + funct3 + rd + opcode}")
    
    def __str__(self) -> str:
        return super().__str__() + f"({self.cmd.cmd}:[{self.rd},{self.rs1},{self.rs2}]"

class ImmediateInstruction(RiscVInstruction):
    required_args: tuple[str, ...] = ("cmd", "rd", "rs1", "imm")
    cmd: OPCode     # funct3 in here
    rd: Register
    rs1: Register
    imm: int | Label
    
    def __init__(self, cmd: str, rd: str, rs1: str, imm: int | Label):
        super().__init__(type=RiscVInstructionType.I)
        self.cmd = OPCode(cmd)
        self.rd = Register(RiscVTokenType.RD, RiscVRegisters[rd])
        self.rs1 = Register(RiscVTokenType.RS1, RiscVRegisters[rs1])
        self.imm = int(imm)
    
    def to_binary(self) -> str:
        opcode, funct3 = self.cmd.to_binary().split(",")
        rd: str = self.rd.to_binary()
        rs1: str = self.rs1.to_binary()
        
        l: str
        if isinstance(self.imm, Label):
            l = self.imm.to_binary()
        else: 
            l = bin_str(self.imm, 12)[::-1]
            
        return l + rs1 + funct3 + rd + opcode
    
    def inspect(self):
        super().inspect()
        
        self.cmd.inspect()
        opcode, funct3 = self.cmd.to_binary().split(",")
        
        self.rd.inspect()
        rd: str = self.rd.to_binary()
        
        self.rs1.inspect()
        rs1: str = self.rs1.to_binary()
        
        l: str
        if isinstance(self.imm, Label):
            l =self.imm.to_binary()
        else: 
            l = bin_str(self.imm, 12)[::-1]
        print(f"imm : {self.imm} = {l}")
        
        print(f"total -> {l + rs1 + funct3 + rd + opcode}")
    
    def __str__(self) -> str:
        return super().__str__() + f"({self.cmd.cmd}:[{self.rd},{self.rs1},{self.imm}]"


class StoreInstruction(RiscVInstruction):
    required_args: tuple[str, ...] = ("cmd", "imm", "rs1", "rs2")
    cmd: OPCode     # funct3 in here
    imm: int
    rs1: Register
    rs2: Register
    
    def __init__(self, cmd: str, rs1: str, rs2: str, imm: int):
        super().__init__(type=RiscVInstructionType.S)
        self.cmd = OPCode(cmd)
        self.rs1 = Register(RiscVTokenType.RS1, RiscVRegisters[rs1])
        self.rs2 = Register(RiscVTokenType.RS2, RiscVRegisters[rs2])
        self.imm = int(imm)
    
    def to_binary(self) -> str:
        opcode, funct3 = self.cmd.to_binary().split(",")
        rs1: str = self.rs1.to_binary()
        rs2: str = self.rs2.to_binary()
        
        l: str = bin_str(self.imm, 12)
            
        return l[10:3:-1] + rs2 + rs1 + funct3 + l[4::-1] + opcode
    
    def inspect(self):
        super().inspect()
        
        self.cmd.inspect()
        opcode, funct3 = self.cmd.to_binary().split(",")
        
        self.rs1.inspect()
        rs1: str = self.rs1.to_binary()
        
        self.rs2.inspect()
        rs2: str = self.rs2.to_binary()
        
        l: str = bin_str(self.imm, 12)
        print(f"imm : {self.imm} = {l[10:4:-1]} {l[4::-1]}")
        
        print(f"total -> {l[10:3:-1] + rs2 + rs1 + funct3 + l[4::-1] + opcode}")
    
    def __str__(self) -> str:
        return super().__str__() + f"({self.cmd.cmd}:[{self.imm},{self.rs1},{self.rs2}]"


class BranchInstruction(RiscVInstruction):
    required_args: tuple[str, ...] = ("cmd", "imm", "rs1", "rs2")
    cmd: OPCode     # funct3 in here
    imm: int | Label
    rs1: Register
    rs2: Register
    
    def __init__(self, cmd: str, rs1: str, rs2: str, imm: int | Label):
        super().__init__(type=RiscVInstructionType.B)
        self.cmd = OPCode(cmd)
        self.rs1 = Register(RiscVTokenType.RS1, RiscVRegisters[rs1])
        self.rs2 = Register(RiscVTokenType.RS2, RiscVRegisters[rs2])
        self.imm = int(imm)
    
    def to_binary(self) -> str:
        opcode, funct3 = self.cmd.to_binary().split(",")
        rs1: str = self.rs1.to_binary()
        rs2: str = self.rs2.to_binary()
        
        
        l: str
        if isinstance(self.imm, Label):
            l = bin_str(self.imm.pc >> 1, 11)
        else: 
            l = bin_str(self.imm >> 1, 11)
            
        return "0" + l[9:3:-1] + rs2 + rs1 + funct3 + l[3::-1] + l[10] + opcode
    
    def inspect(self):
        super().inspect()
        
        self.cmd.inspect()
        opcode, funct3 = self.cmd.to_binary().split(",")
        
        self.rs1.inspect()
        rs1: str = self.rs1.to_binary()
        
        self.rs2.inspect()
        rs2: str = self.rs2.to_binary()
        
        
        l: str
        if isinstance(self.imm, Label):
            l = bin_str(self.imm.pc >> 1, 11)
        else:
            l = bin_str(self.imm >> 1, 11)
        
        print(f"imm : {self.imm} = 0 {l[9:3:-1]} {l[3::-1]} {l[10]}")
        
        print(f"total -> 0{l[9:3:-1] + rs2 + rs1 + funct3 + l[3::-1] + l[10] + opcode}")
    
    def __str__(self) -> str:
        return super().__str__() + f"({self.cmd.cmd}:[{self.imm},{self.rs1},{self.rs2}]"


class UpperImmediateInstruction(RiscVInstruction):
    required_args: tuple[str, ...] = ("cmd", "rd", "imm")
    cmd: OPCode
    rd: Register
    imm: int | Label
    
    def __init__(self, cmd: str, rd: str, imm: int | Label):
        super().__init__(type=RiscVInstructionType.U)
        self.cmd = OPCode(cmd)
        self.rd = Register(RiscVTokenType.RD, RiscVRegisters[rd])
        self.imm = int(imm)
    
    def to_binary(self) -> str:
        opcode = self.cmd.to_binary().split(",")[0]
        rd: str = self.rd.to_binary()
        
        l: str
        if isinstance(self.imm, Label):
            l = self.imm.to_binary()
        else: 
            l = bin_str(self.imm, 12)[::-1]
            
        return l + rd + opcode
    
    def inspect(self):
        super().inspect()
        
        self.cmd.inspect()
        opcode = self.cmd.to_binary().split(",")[0]
        
        self.rd.inspect()
        rd: str = self.rd.to_binary()
        
        l: str
        if isinstance(self.imm, Label):
            l =self.imm.to_binary()
        else: 
            l = bin_str(self.imm, 12)[::-1]
        print(f"imm : {self.imm} = {l}")
        
        print(f"total -> {l + rd + opcode}")
    
    def __str__(self) -> str:
        return super().__str__() + f"({self.cmd.cmd}:[{self.rd},{self.imm}]"


class JumpInstruction(RiscVInstruction):
    required_args: tuple[str, ...] = ("cmd", "rd", "imm")
    cmd: OPCode
    rd: Register
    imm: int | Label
    
    def __init__(self, cmd: str, rd: str, imm: int | Label):
        super().__init__(type=RiscVInstructionType.J)
        self.cmd = OPCode(cmd)
        self.rd = Register(RiscVTokenType.RD, RiscVRegisters[rd])
        self.imm = int(imm)
    
    def to_binary(self) -> str:
        opcode = self.cmd.to_binary().split(",")[0]
        rd: str = self.rd.to_binary()
        
        l: str
        if isinstance(self.imm, Label):
            l = bin_str(self.imm.pc >> 1, 19)
        else: 
            l = bin_str(self.imm >> 1, 19)
            
        return "0" + l[9::-1] + l[10] + l[18:10:-1] + rd + opcode
    
    def inspect(self):
        super().inspect()
        
        self.cmd.inspect()
        opcode = self.cmd.to_binary().split(",")[0]
        
        self.rd.inspect()
        rd: str = self.rd.to_binary()
        
        l: str
        if isinstance(self.imm, Label):
            l =self.imm.to_binary()
        else: 
            l = bin_str(self.imm, 19)[::-1]
        print(f"imm : {self.imm} = 0 {l[9::-1]} {l[10]} {l[18:11:-1]}")
        
        print(f"total -> 0{l[9::-1] + l[10] + l[18:11:-1] + rd + opcode}")
    
    def __str__(self) -> str:
        return super().__str__() + f"({self.cmd.cmd}:[{self.rd},{self.imm}]"


InstructionConstructorDictionary: dict[RiscVInstructionType, Type[RiscVInstruction]] = {
    RiscVInstructionType.R: RegisterInstruction
}


MEM_ACCESS: re.Pattern = re.compile(r"([A-Za-z0-9]+)\(([A-Za-z0-9]+)\)")
GUARD: object = object()

def parse_instruction(cmd: str, labels: dict[str, Label], toks: list[str]) -> RiscVInstruction:
    """Parse instruction from token leftovers.

    Args:
        cmd (str): Command part of this instruction. This should be parsed first to determine argument structure.
        labels (dict[str, int]): Markers in current code.
        toks (list[str]): Token leftovers from code.

    Returns:
        RiscVInstruction: Parsed instruction object.
    """
    type: RiscVInstructionType = InstructionTypeDictionary[cmd]
    
    parsed_args: list[str | Label | int] = [*toks, cast(int, GUARD)]   # len(toks) + 1 의 길이.
    
    for i, tok in enumerate(toks):
        if tok in labels:
            parsed_args[i] = labels[tok]
        m = MEM_ACCESS.match(tok)
        if m is not None:
            offset, reg = m.groups()
            parsed_args[i] = reg
            parsed_args[i+1] = offset
    
    if parsed_args[-1] is GUARD:
        del parsed_args[-1]

    match type:
        case RiscVInstructionType.R:
            # Register type
            return RegisterInstruction(cmd, *cast(tuple[str, str, str], tuple(parsed_args)))
            
        case RiscVInstructionType.I:
            # Immediate type
            return ImmediateInstruction(cmd, *cast(tuple[str, str, int | Label], tuple(parsed_args)))
            
        case RiscVInstructionType.S:
            # Immediate type
            return StoreInstruction(cmd, *cast(tuple[str, str, int], tuple(parsed_args)))
            
        case RiscVInstructionType.B:
            # Immediate type
            return BranchInstruction(cmd, *cast(tuple[str, str, int], tuple(parsed_args)))
        
        case RiscVInstructionType.U:
            # Immediate type
            return UpperImmediateInstruction(cmd, *cast(tuple[str, int | Label], tuple(parsed_args)))
        
        case RiscVInstructionType.J:
            # Immediate type
            return JumpInstruction(cmd, *cast(tuple[str, int | Label], tuple(parsed_args)))
        
        case _:
            # should not reach here
            raise TypeError("Unknown RISC-V instruction type!")
