

from pprint import pprint
from .instructions import Label, RiscVInstruction, parse_instruction


class RiscVProgram:
    """
    Parser for entire RISC-V program.
    
    Steps:
    1. Create a new RiscVProgram instance.
    ex) p = RiscVProgram("code...")
    2. parse it.
    ex) p.parse()
    3. see hexademical representations of each lines:
    ex) p.show_hex()
    """
    code: str
    instructions: list[RiscVInstruction]
    
    
    def __init__(self, code: str):
        self.code = code.strip("\n")
        pprint(self.code)
        self.labels: dict[str, Label] = {}

    def parse(self):
        self.instructions = []
        
        lines = self.code.splitlines()
        
        pc: int = 0
        for line in lines:
            if not line.startswith(" ") and line.endswith(":"):
                print(line)
                # markers
                label = line.rstrip(":")
                self.labels[label] = Label(label, pc)
            pc += 4
        
        print("Labels in current RISC-V code:")
        pprint(self.labels)
        print()
        
        pc = 0
        for line in lines:
            print(f"pc={pc:02d} -> {line}")
            if line == "" or line.endswith(":"):
                continue
            cmd: str = ""
            line = line.lstrip(" ")
            for c in line:
                if c == " ":
                    break
                cmd += c

            toks: list[str] = line[len(cmd):].lstrip().split(", ")
            inst: RiscVInstruction = parse_instruction(cmd, self.labels, toks)
            
            self.instructions.append(inst)
            pc += 4
        print("===")
    
    def check(self):
        for inst in self.instructions:
            print(inst.to_binary())
            assert len(inst.to_binary()) == 32
    
    def show_bin(self):
        pc: int = 0
        
        for inst in self.instructions:
            print(f"{pc:02d} : {inst.to_binary()}")
            pc += 4
    
    def show_hex(self):
        pc: int = 0
        
        for inst in self.instructions:
            print(f"{pc:02d} : {inst.to_hex()}")
            pc += 4
    
    def inspect_all(self):
        pc: int = 0
        
        for inst in self.instructions:
            print(f"{pc:02d} : ", end="")
            inst.inspect()
            assert len(inst.to_binary()) == 32
            pc += 4
