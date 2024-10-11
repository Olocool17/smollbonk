from __future__ import annotations
import click
import re
from pathlib import Path
import string
import itertools
from typing import Generator, Callable
from io import BufferedReader
from copy import deepcopy
import time

source_extension = ".sbonk"
bonk_extension = ".bonk"


class Parser:
    def __init__(self, source: BufferedReader):
        self.source = source
        self.ident = 0
        self.line_count = 0
        self.node = Block()

    def add(self, new_node) -> Node:
        new_node.parent = self.node
        self.node.children.append(new_node)
        return new_node

    def add_and_indent(self, new_node) -> None:
        new_node = self.add(new_node)
        self.node = new_node
        self.ident += 1

    def dedent(self) -> None:
        self.node = self.node.parent
        self.ident -= 1

    def parse(self) -> Block:
        try:
            return self._parse()
        except Exception as e:
            e.add_note(f"... while parsing '{self.source.name}:{self.line_count}'")
            raise

    def _parse(
        self,
    ) -> Block:
        while True:
            if (
                self.source.peek(self.ident * 4)[: self.ident * 4].decode().count(" ")
                < self.ident * 4
            ):
                self.dedent()
                continue

            raw_line = self.source.readline().decode()
            if len(raw_line) == 0:
                return self.node
            self.line_count += 1

            line = raw_line.strip()
            if len(line) == 0:
                continue

            match = re.match(
                r"repeat +([0-9]+) *:(.*)",
                line,
            )
            if match is not None:
                self.add_and_indent(RepeatBlock(int(match.group(1))))
                if match.group(2) is not None and len(match.group(2).strip()) > 0:
                    self.add(self.parse_instruction(match.group(2)))
                continue

            match = re.match(
                r"(?:template<([a-zA-Z0-9_ ,]+)> *)?(label|proc) +([a-zA-Z0-9_]+) *:(.*)",
                line,
            )
            if match is not None:
                if match.group(2) == "label":
                    cls = Label
                elif match.group(2) == "proc":
                    cls = Procedure
                if match.group(1) is not None:
                    template_args: tuple[str] = tuple(
                        e.strip() for e in match.group(1).split(",")
                    )
                    self.add_and_indent(Template(match.group(3), cls, template_args))
                else:
                    self.add_and_indent(cls(match.group(3)))
                if match.group(4) is not None and len(match.group(4).strip()) > 0:
                    self.add(self.parse_instruction(match.group(4)))
                continue

            match = re.match(r"template<([a-zA-Z0-9_\* ,]+)> *:(.*)", line)
            if match is not None and match.group(1) is not None:
                literal_args: tuple[str] = tuple(
                    e.strip() for e in match.group(1).split(",")
                )
                self.add_and_indent(
                    TemplateSpecialisationBlock(literal_args=literal_args)
                )
                if match.group(2) is not None and len(match.group(2).strip()) > 0:
                    self.add(self.parse_instruction(match.group(2)))
                continue

            if line[:2] == "--":
                continue
            self.add(self.parse_instruction(line))

    @staticmethod
    def parse_instruction(line: str) -> Instruction:
        parts = [e.strip() for e in line.strip().split()]
        ins = parts[0]
        args_str = " ".join(parts[1:])
        if ins not in instructions:
            raise CompileError(f"Instruction '{ins}' is not a valid instruction.")
        ins_type, arg_types = instructions[ins]

        ins = ins_type(ins)
        for arg_type in arg_types:
            if len(args_str.strip()) == 0:
                raise CompileError(
                    f"Instruction '{ins.ins}' requires {len(arg_types)} arguments but fewer were given."
                )
            match = re.match(arg_type.re_pattern, args_str)
            if match is None:
                raise CompileError(
                    f"'{args_str} could not be parsed to a valid {arg_type.__name__}."
                )
            ins.args.append(arg_type(ins, match.group(0)))
            args_str = args_str[match.end() :].lstrip()
        if len(args_str.strip()) > 0:
            raise CompileError(
                f"Instruction '{ins.ins}' requires {len(arg_types)} arguments but more were given."
            )
        return ins


class Node:
    def __init__(self):
        self.parent: Block | None = None

    def cut(self):
        self.parent.children.remove(self)
        self.parent = None

    def insert_after(self, after : Node):
        after.parent.children.insert(after.parent.children.index(after) + 1, self)
        self.parent = after.parent

    def deepcopy_tree(self):
        parent = self.parent 
        self.parent = None
        copy = deepcopy(self)
        self.parent = parent
        return copy
        

    def __str__(self):
        return self.__class__.__name__


class Block(Node):
    def __init__(self):
        super().__init__()
        self.children: list[Node] = []

    def transplant_children(self, after : Node):
        i = after.parent.children.index(after)
        for c in self.children:
            i += 1
            after.parent.children.insert(i, c)
            c.parent = after.parent
        transplanted_children = self.children
        self.children = []
        return transplanted_children

    def copy_children(self, after : Node) -> list[Node]:
        i = after.parent.children.index(after)
        copies_list = []
        for c in self.children:
            i += 1
            c_copy = c.deepcopy_tree()
            copies_list.append(c_copy)
            after.parent.children.insert(i, c_copy)
            c_copy.parent = after.parent
        return copies_list
    
    def copy_children_on(self, on : Block):
        for c in self.children:
            c_copy = c.deepcopy_tree()
            c_copy.parent = on
            on.children.append(c_copy)

    def emit_children(self, indent):
        s = ""
        indent_string = "\n"
        for _ in range(indent):
            indent_string += "    "
        for n in self.children:
            emission = n.emit()
            if len(emission.strip()) == 0:
                continue
            s += indent_string
            s += indent_string.join(emission.split("\n"))
        return s + "\n"
    
    def recurse(self, func: str, *args, **kwargs):
        for c in [c for c in self.children]:
            if hasattr(c, func):
                getattr(c, func)(*args, **kwargs)
            else:
                c.recurse(func, *args, **kwargs)


    def __str__(self):
        s = f"{self.__class__.__name__}"
        for n in self.children:
            s += "\n    "
            s += "\n    ".join(str(n).split("\n"))
        return s + "\n"


class Template(Block):
    registry: dict[str, Template] = {}

    def __init__(self, name: str, cls: type[Block], template_args: tuple):
        super().__init__()
        self.name = name
        self.cls = cls
        self.template_args = template_args
        self.instantiations: dict[tuple, TemplateableBlock] = {}

    def __getitem__(self, literal_args: tuple) -> TemplateableBlock:
        return self.instantiations[literal_args]
    
    def stamp(self, template_translation: dict[str, str]):
        pass
    
    def register_labels(self):
        self.registry[self.name] = self

    def register_targets(self):
        pass


    def add_instantiation(self, literal_args: tuple): 
        instantiation = self.cls(self.name)
        for c in self.children:
            c.parent = None
        instantiation.children = deepcopy(self.children)
        for c in self.children:
            c.parent = self
        for c in instantiation.children:
            c.parent = instantiation
        instantiation.parent = self.parent
        template_translation = {k: v for k, v in zip(self.template_args, literal_args)}
        instantiation.stamp_root(template_translation)
        self.instantiations[literal_args] = instantiation

    @staticmethod
    def remove_internals(template_translation: dict[str, str]) -> dict[str, str]:
        return {k:v for k,v in template_translation.items() if not k.startswith("__")}
    
    def emit(self) -> str:
        s = ""
        indent_string = "\n"
        for instantiation in self.instantiations.values():
            emission = instantiation.emit()
            if len(emission.strip()) == 0:
                continue
            s += indent_string
            s += indent_string.join(emission.split("\n"))
        return s + "\n"

    def __str__(self):
        s = f"{self.__class__.__name__}:{self.cls.__name__}({self.name})"
        for n in self.children:
            s += "\n    "
            s += "\n    ".join(str(n).split("\n"))
        return s + "\n"


class RepeatBlock(Block):
    def __init__(self, repeat_count: int):
        super().__init__()
        self.repeat_count: int = repeat_count

    def stamp(self, template_translation: dict[str, str]):
        for _ in range(self.repeat_count):
            for copy_c in self.copy_children(self):
                copy_c.stamp(template_translation)
        self.cut()


class TemplateableBlock(Block):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __str__(self):
        s = f"{self.__class__.__name__}({self.name})"
        for n in self.children:
            s += "\n    "
            s += "\n    ".join(str(n).split("\n"))
        return s + "\n"
    
class Label(TemplateableBlock):
    registry : dict[Label] = {}

    def stamp_root(self, template_translation: dict[str, str]):
        self.stamp(template_translation)
        self.recurse("stamp", template_translation)
        self.recurse("register_labels")
        self.recurse("register_targets")

    def stamp(self, template_translation: dict[str, str]):
        if len(template_translation) > 0:
            self.mangled_name = self.name + "__" + "__".join(v for v in template_translation.values())
        else:
            self.mangled_name =  self.name
        self.recurse("stamp", template_translation)

    def register_labels(self):
        Label.registry[self.name] = self
        self.recurse("register_labels")

    def register_targets(self):
        Label.registry[self.name] = self
        self.recurse("register_targets")


    def emit(self) -> str:
        return f"label {self.mangled_name}:" + self.emit_children(1)


class Procedure(TemplateableBlock):
    proc_counter : int = 0
    registry : dict[Procedure] = {}

    def stamp_root(self, template_translation : dict[str,str]):
        self.stamp(template_translation)

    def stamp(self, template_translation : dict[str, str]):
        self.template_translation = deepcopy(template_translation)
        trailing_label = Label(self.name + "__return")
        self.children.append(trailing_label)
        trailing_label.parent = self

    def register_labels(self):
        Procedure.registry[self.name] = self

    def register_targets(self):
        pass

    def inline(self, after: Node) -> list[Node]:
        self.template_translation["__proc_counter"] = str(self.proc_counter)
        inlined = Block() 
        self.copy_children_on(inlined)
        inlined.recurse("stamp", self.template_translation)
        inlined.recurse("register_labels")
        inlined.recurse("replace_return", inlined.children[-1])
        inlined.recurse("register_targets")
        self.proc_counter += 1
        return inlined.transplant_children(after)

    def emit(self) -> str:
        return ""


class TemplateSpecialisationBlock(Block):
    def __init__(self, literal_args: tuple[str]):
        super().__init__()
        self.literal_args: tuple[str] = literal_args

    def stamp(self, template_translation: dict[str, str]):
        template_translation = Template.remove_internals(template_translation)
        if len(self.literal_args) != len(template_translation):
            raise CompileError(f"Enclosing template has {len(template_translation)} template arguments, but template specialisation provided {len(self.literal_args)}.")
        for literal_arg, template_translation_arg in zip(self.literal_args, template_translation.values()):
            if literal_arg == "*":
                continue
            if literal_arg != template_translation_arg:
                self.cut()
                return
        self.recurse("stamp", template_translation)
        self.transplant_children(self)
        self.cut()

    def register_labels(self):
        pass

    def register_targets(self):
        pass


class Instruction(Node):
    def __init__(self, ins: str):
        super().__init__()
        self.ins: str = ins
        self.args: list[Argument] = []

    def stamp(self, template_translation: dict[str, str]):
        for arg in self.args:
            arg.stamp(template_translation)

    def recurse(self, func: str, *args, **kwargs):
        for arg in self.args:
            try:
                getattr(arg, func)(*args, **kwargs)
            except AttributeError:
                continue

    def __str__(self):
        return f"{self.__class__.__name__}({self.ins})"

    def emit(self) -> str:
        return self.ins + " " + " ".join(arg.emit() for arg in self.args)


class JumpInstruction(Instruction):
    def register_targets(self):
        arg = self.args[0]
        if isinstance(arg, LabelArgument):
            arg.register_targets(Label)

    def emit(self) -> str:
        arg = self.args[0]
        if isinstance(arg, LabelArgument):
            return self.ins + " " + arg.target.mangled_name
        return self.ins + " " + arg.arg


class CallInstruction(JumpInstruction):
    def register_targets(self):
        self.args[0].register_targets(Procedure)     
        proc : Procedure = self.args[0].target
        inlined = proc.inline(self)
        if self.ins in ("calltrue", "callfalse"):
            if self.ins == "calltrue":
                j_ins = JumpInstruction("jfalse")
            if self.ins == "callfalse":
                j_ins = JumpInstruction("jtrue")
            j_ins.insert_after(self)
            j_ins_arg = LabelArgument(j_ins, "")
            j_ins_arg.target = inlined[-1]
            j_ins.args = [j_ins_arg]
        self.cut()

    def emit():
        raise CompileError()


class Argument:
    re_pattern: str = r"\$?(?:[a-zA-Z0-9_]|(?:\[[a-zA-Z0-9_ ]+\]))+"

    def __init__(self, ins: Instruction, arg: str):
        self.ins = ins
        self.arg = arg

    def stamp(self, template_translation: dict[str, str]):
        for k, v in template_translation.items():
            self.arg = re.sub(f"\\[ *{k} *\\]", v, self.arg)

    def emit(self) -> str:
        return self.arg


class ConstantArgument(Argument):
    re_pattern: str = r"(?:0b|0x|)(?:[a-fA-F0-9]|(?:\[[a-zA-Z0-9_ ]+\]))+"


class LabelArgument(Argument):
    re_pattern: str = r"(?:[a-zA-Z0-9_]|(?:\[[a-zA-Z0-9_ ]+\]))+(?:<[a-zA-Z0-9_ ,]+>)?"
    target = None
    def register_targets(self, target_cls : type[TemplateableBlock]):
        if len(self.template_args) == 0:
            if self.arg != "return":
                self.target = target_cls.registry[self.arg]
            if self.target is None:
                raise CompileError(f"No viable block target found for label argument '{self.arg}'.")
            return
        if self.target_name not in Template.registry:
            raise CompileError(f"No viable template found with name '{self.target_name}' for label argument '{self.arg}'.")
        target_template = Template.registry[self.target_name]
        if target_template.cls is not target_cls:
            raise CompileError(f"Template {target_template.name} produces {target_template.cls}, but one that produces {target_cls} is required.")
        if len(self.template_args) != len(target_template.template_args):
            raise CompileError(f"Template {target_template.name} requires {len(target_template.template_args)} template arguments but {len(self.template_args)} were provided.")
        if self.template_args not in target_template.instantiations:
            target_template.add_instantiation(self.template_args)
        self.target = target_template[self.template_args]


    def stamp(self, template_translation: dict[str, str]):
        Argument.stamp(self, template_translation)
        match = re.search(r"<([a-zA-Z0-9_ ,]+)>", self.arg)
        if match is None:
            self.template_args = tuple()
            return
        template_args = tuple(e.strip() for e in  match.group(1).split(","))
        template_translation = Template.remove_internals(template_translation)
        self.template_args = tuple(template_translation[t] if t in template_translation else t for t in template_args)
        self.target_name = self.arg[:match.start()]

    def replace_return(self, trailing_label : Label):
        if self.arg != "return":
            return
        self.target = trailing_label


class RegisterArgument(Argument):
    re_pattern: str = r"\$(?:[a-zA-Z0-9_]|(?:\[[a-zA-Z0-9_ ]+\]))+"


instructions: dict[str, tuple[type[Instruction], tuple[type[Argument], ...]]] = {
    "add": (Instruction, (Argument, Argument, RegisterArgument)),
    "sub": (Instruction, (Argument, Argument, RegisterArgument)),
    "mul": (Instruction, (Argument, Argument, RegisterArgument)),
    "div": (Instruction, (Argument, Argument, RegisterArgument)),
    "mod": (Instruction, (Argument, Argument, RegisterArgument)),
    "rand": (Instruction, (Argument, Argument, RegisterArgument)),
    "and": (Instruction, (Argument, Argument, RegisterArgument)),
    "or": (Instruction, (Argument, Argument, RegisterArgument)),
    "xor": (Instruction, (Argument, Argument, RegisterArgument)),
    "not": (Instruction, (Argument, RegisterArgument)),
    "lshift": (Instruction, (Argument, Argument, RegisterArgument)),
    "rshift": (Instruction, (Argument, Argument, RegisterArgument)),
    "mov": (Instruction, (Argument, RegisterArgument)),
    "read": (Instruction, (Argument, RegisterArgument)),
    "write": (Instruction, (Argument, Argument)),
    "cmp": (Instruction, (Argument, Argument)),
    "gte": (Instruction, (Argument, Argument)),
    "lte": (Instruction, (Argument, Argument)),
    "gt": (Instruction, (Argument, Argument)),
    "lt": (Instruction, (Argument, Argument)),
    "jmp": (JumpInstruction, (LabelArgument,)),
    "jtrue": (JumpInstruction, (LabelArgument,)),
    "jfalse": (JumpInstruction, (LabelArgument,)),
    "jumpir": (JumpInstruction, (Argument,)),
    "call": (CallInstruction, (LabelArgument,)),
    "calltrue": (CallInstruction, (LabelArgument,)),
    "callfalse": (CallInstruction, (LabelArgument,)),
    "wait": (Instruction, ()),
    "stop": (Instruction, ()),
    "syscall": (Instruction, ()),
    "log": (Instruction, (Argument,)),
    "fwd": (Instruction, ()),
    "turnl": (Instruction, (Argument,)),
    "turnr": (Instruction, (Argument,)),
    "fsens": (Instruction, ()),
    "lsens": (Instruction, ()),
    "rsens": (Instruction, ()),
    "nsens": (Instruction, ()),
}


class CompileError(Exception):
    pass


def compile(source: BufferedReader) -> str:
    print("Parsing...", end="")
    begin_time = time.time_ns()
    root = Parser(source).parse()
    end_time = time.time_ns()
    print(f" took {(end_time - begin_time) / 1000000} ms.")
    begin_time = end_time

    print(f"AST for '{source.name}':")
    print(root)
    print("Compiling...", end="")
    root.recurse("stamp", {})
    root.recurse("register_labels")
    root.recurse("register_targets")
    end_time = time.time_ns()
    print(f" took {(end_time - begin_time) / 1000000} ms.")
    begin_time = end_time

    print("Emitting...", end="")
    emission =  root.emit_children(0)
    end_time = time.time_ns()
    print(f" took {(end_time - begin_time) / 1000000} ms.")
    begin_time = end_time
    return emission


static_registers = (
    "compare_register",
    "health",
    "tick",
    "syscall_register",
    "syscall_arg_register",
    "syscall_sens_register",
    "br1",
    "br2",
    "br3",
    "br4",
    "br5",
    "br6",
    "br7",
    "br8",
    "edge_nearby",
    "wall_nearby",
    "player_nearby",
    "powerup_nearby",
)
register_chars = string.ascii_lowercase + string.digits + "_"

static_labels = "main"


def shortest_generator(chars: str, max_len: int = 8) -> Generator[int, None, None]:
    products_length = 1
    products = itertools.product(chars, repeat=products_length)
    while products_length <= max_len:
        try:
            yield "".join(next(products))
        except StopIteration:
            products_length += 1
            products = itertools.product(chars, repeat=products_length)


def minify(source: str) -> str:
    # rename registers
    registers: dict = {}
    gen = shortest_generator(register_chars)
    for register in re.finditer(r"\$([a-zA-Z0-9_]+)", source):
        register = register.group(1)
        if register in static_registers:
            continue
        if register not in registers:
            registers[register] = next(gen)

    for k, v in registers.items():
        print(f"register {k} -> {v}")
        source = re.sub(f"\\${k}(\\s)", f"${v}\\1", source)

    # rename labels
    labels: dict = {}
    gen = shortest_generator(register_chars)
    for label in re.finditer(r"label +([a-zA-Z0-9_]+)", source):
        label = label.group(1)
        if label in static_labels:
            continue
        if label not in labels:
            labels[label] = next(gen)

    for k, v in labels.items():
        print(f"label {k} -> {v}")
        source = re.sub(f"(jmp|jtrue|jfalse) +{k}(\\s)", f"\\1 {v}\\2", source)
        source = re.sub(f"label +{k} *:", f"label {v}:", source)

    # strip whitespace
    source = source.split("\n")
    minified: list[str] = [line.strip() for line in source if len(line.strip()) > 0]
    return "\n".join(minified)


@click.command("compile")
@click.argument("input", type=click.Path(exists=True), nargs=1)
@click.argument("output", type=click.Path(), required=False, default=None, nargs=1)
@click.option("-m", "--mini", is_flag=True, default=False)
def compile_command(input: str, output: str | None, mini: bool) -> None:
    input: Path = Path(input)
    if output is None:
        output: Path = input.with_suffix(bonk_extension)
        if input.name == output.name:
            print(
                f"Avoid using the extension '{bonk_extension}' as a source extension. For example, use '{source_extension}' instead"
            )
            output = output.parent / (
                Path(str(input.with_suffix("")) + "-compiled" + bonk_extension)
            )
    else:
        output: Path = Path(output)

    compiled: str = None
    with input.open("rb") as infile:
        compiled = compile(infile)
    if mini:
        compiled = minify(compiled)
    with output.open("w") as outfile:
        outfile.write(compiled)


@click.command("minify")
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path(), required=False, default=None, nargs=1)
def minify_command(input: str, output: str | None) -> None:
    input: Path = Path(input)
    if output is None:
        output: Path = Path(str(input.with_suffix("")) + "-min" + bonk_extension)
    else:
        output: Path = Path(output)
    minified: str = None
    with input.open("r") as infile:
        minified = minify(infile.read())
    with output.open("w") as outfile:
        outfile.write(minified)


@click.group()
def cli():
    pass


cli.add_command(compile_command)
cli.add_command(minify_command)

if __name__ == "__main__":
    cli()
