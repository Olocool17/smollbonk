import click
import re
from pathlib import Path
import string
import itertools
from typing import Generator, Callable
from io import BufferedReader

source_extension = ".sbonk"
bonk_extension = ".bonk"


class Node:
    def __init__(self):
        self.parent: Block | None = None

    def __str__(self):
        return self.__class__.__name__


class Block(Node):
    def __init__(self):
        super().__init__()
        self.children: list[Node] = []

    def add(self, cls, *args, **kwargs) -> "Node":
        new_node = cls(*args, **kwargs)
        new_node.parent = self
        self.children.append(new_node)
        return new_node

    def parse(self, source: BufferedReader, ident: int = 0, trailing_line: str = None):
        while True:
            if trailing_line is not None:
                line = trailing_line.strip()
                trailing_line = None
            else:
                if source.peek(ident * 4)[: ident * 4].decode().count(" ") < ident * 4:
                    return
                raw_line = source.readline().decode()
                if len(raw_line) == 0:
                    return
                line = raw_line.strip()

            if len(line) == 0:
                continue

            match = re.match(
                r"repeat +([0-9]+) *:(.*)",
                line,
            )
            if match is not None:
                self.add(RepeatBlock, int(match.group(1))).parse(
                    source, ident=ident + 1, trailing_line=match.group(2)
                )
                continue

            match = re.match(
                r"(?:template<([a-zA-Z0-9_ ,]+)> *)?(label|proc) +([a-zA-Z0-9_]+) *:(.*)",
                line,
            )
            if match is not None:
                template_args = tuple()
                if match.group(1) is not None:
                    template_args: tuple[str] = tuple(
                        e.strip() for e in match.group(1).split(",")
                    )
                if match.group(2) == "label":
                    cls = Label
                elif match.group(2) == "proc":
                    cls = Procedure
                self.add(cls, match.group(3), template_args=template_args).parse(
                    source, ident=ident + 1, trailing_line=match.group(4)
                )
                continue

            match = re.match(r"template<([a-zA-Z0-9_\* ,]+)> *:(.*)", line)
       
            if match is not None:
                template_args = tuple()
                if match.group(1) is not None:
                    template_args: tuple[str] = tuple(
                        e.strip() for e in match.group(1).split(",")
                    )
                self.add(
                    TemplateSpecialisationBlock, template_args=template_args
                ).parse(source, ident=ident + 1, trailing_line=match.group(2))
                continue

            ins = line.split()[0]
            args = " ".join(line.split()[1:])
            if ins in ("jmp", "jtrue", "jfalse"):
                self.add(JumpInstruction, ins, args)
                continue
            if ins in ("call", "calltrue", "callfalse"):
                self.add(CallInstruction, ins, args)
                continue
            if ins in (
                "add",
                "sub",
                "mul",
                "div",
                "mod",
                "rand",
                "and",
                "or",
                "xor",
                "not",
                "lshift",
                "rshift",
                "mov",
                "read",
                "write",
                "cmp",
                "gte",
                "lte",
                "gt",
                "lt",
                "jumpir",
                "wait",
                "stop",
                "syscall",
                "log",
                "fwd",
                "turnl",
                "turnr",
                "fsens",
                "lsens",
                "rsens",
                "nsens",
            ):
                self.add(Instruction, ins, args)
                continue
            print(f"Line '{line}' could not be parsed.")

    def recurse(self, func: Callable[[Node], None], *args, **kwargs):
        for c in self.children:
            try:
                getattr(c, func.__name__)(*args, **kwargs)
            except AttributeError:
                if isinstance(c, Block):
                    c.recurse(func, *args, **kwargs)

    def emit_children(self, indent, **kwargs):
        s = ""
        indent_string = "\n"
        for _ in range(indent):
            indent_string += "    "
        for n in self.children:
            emission = n.emit(**kwargs)
            if len(emission.strip()) == 0:
                continue
            s += indent_string
            s += indent_string.join(emission.split("\n"))
        return s + "\n"

    def __str__(self):
        s = f"{self.__class__.__name__}"
        for n in self.children:
            s += "\n    "
            s += "\n    ".join(str(n).split("\n"))
        return s + "\n"


class RepeatBlock(Block):
    def __init__(self, repeat_count: int):
        super().__init__()
        self.repeat_count: int = repeat_count

    def emit(self, **kwargs):
        s = ""
        for _ in range(self.repeat_count):
            s += self.emit_children(0, **kwargs)
        return s


class TemplatableBlock(Block):
    registry: dict[str, "TemplatableBlock"] = {}

    def __init__(self, name: str, template_args: tuple[str]):
        super().__init__()
        self.name: str = name
        if name in TemplatableBlock.registry:
            raise CompileError(f"Duplicate block identifier '{name}'.")
        TemplatableBlock.registry[name] = self
        self.template_args: tuple[str] = template_args
        self.instantiations: set[tuple[str]] = set()
        if len(template_args) == 0:
            self.instantiations.add(tuple())

    def get_literal_args(self, **kwargs) -> tuple[str]:
        for template_arg in self.template_args:
            if template_arg not in kwargs:
                raise CompileError()
        return tuple(kwargs[template_arg] for template_arg in self.template_args)

    def get_kwargs(self, instantiation: tuple[str]) -> dict[str, str]:
        if len(instantiation) != len(self.template_args):
            raise CompileError()
        return {k: v for k, v in zip(self.template_args, instantiation)}

    def get_literal_target(self, instantiation: tuple[str], **kwargs) -> str:
        if instantiation not in self.instantiations:
            raise CompileError()
        target = self.name
        if (len(self.template_args) == 0):
            for v in kwargs.values():
                target += "__" + str(v)
        for literal_arg in instantiation:
            target += "__" + literal_arg
        return target

    def register_target(self, **kwargs):
        if len(self.template_args) != 0:
            return
        else:
            self.recurse(JumpInstruction.register_target, **kwargs)

    def add_instantiation(self, instantiation: tuple[str]):
        if instantiation in self.instantiations:
            return
        self.instantiations.add(instantiation)
        self.recurse(
            JumpInstruction.register_target,
            **self.get_kwargs(instantiation),
        )


class Label(TemplatableBlock):
    def emit(self, **kwargs) -> str:
        s = ""
        for instantiation in self.instantiations:
            s += (
                f"label {self.get_literal_target(instantiation, **kwargs)}:"
                + self.emit_children(1, **kwargs, **self.get_kwargs(instantiation))
            )
        return s


class Procedure(TemplatableBlock):
    def __init__(self, name: str, template_args: tuple[str]):
        super().__init__(name, template_args)
        self.counters = {k: 0 for k in self.instantiations}
    def add_instantiation(self, instantiation: tuple[str]):
        TemplatableBlock.add_instantiation(self, instantiation)
        self.counters = {k: 0 for k in self.instantiations}

    def get_return_target(self, instantiation : tuple[str]) -> str:
        return TemplatableBlock.get_literal_target(self, instantiation)+ f"__{self.counters[instantiation]}"

    def inline(self, instantiation: tuple[str]) -> str:
        if instantiation not in self.instantiations:
            raise CompileError
        target = (
            self.get_return_target(instantiation) 
        )
        s = self.emit_children(
            0, **self.get_kwargs(instantiation), counter=self.counters[instantiation]
        )
        s += f"label {target}:\n"
        self.counters[instantiation] += 1
        return s

    def emit(self, **kwargs) -> str:
        return ""


class TemplateSpecialisationBlock(Block):
    def __init__(self, template_args: tuple[str]):
        super().__init__()
        self.literal_args: tuple[str] = template_args

    def check_literal_args(self, **kwargs) -> bool:
        parent = self.parent
        while parent is not None:
            if isinstance(parent, TemplatableBlock):
                if len(parent.template_args) == 0:
                    parent = parent.parent
                    continue
                if len(parent.template_args) != len(self.literal_args):
                    raise CompileError(
                        "Template specialisation blocks must have the exact same number of arguments as their parent template blocks."
                    )
                for template_arg, literal_arg in zip(
                    parent.template_args, self.literal_args
                ):
                    if literal_arg == "*":
                        continue    
                    if kwargs[template_arg] != literal_arg:
                        return False
                return True
            parent = parent.parent

    def register_target(self, **kwargs):
        if not self.check_literal_args(**kwargs):
            return
        self.recurse(JumpInstruction.register_target, **kwargs)

    def emit(self, **kwargs) -> str:
        if not self.check_literal_args(**kwargs):
            return ""
        return self.emit_children(0, **kwargs)


class Instruction(Node):
    def __init__(self, ins: str, args: str):
        super().__init__()
        self.ins: str = ins
        self.args: str = args

    def emit(self, **kwargs) -> str:
        return f"{self.ins} {self.args}"


class JumpInstruction(Instruction):
    def register_target(self, **kwargs):
        match = re.match(r"([a-zA-Z0-9_]+)(?:<([a-zA-Z0-9_ ,]+)>)?", self.args)
        self.target_name = match.group(1)
        if self.target_name == "return":
            return
        if self.target_name not in TemplatableBlock.registry:
            raise CompileError(
                f"No block identifier found with name '{self.target_name}'."
            )
        self.target_template_args = tuple()
        if match.group(2) is not None:
            self.target_template_args = tuple(
                e.strip() for e in match.group(2).split(",")
            )
            # if applicable: replace template args with literals
            target_literal_args = tuple(
                kwargs[arg] if arg in kwargs else arg
                for arg in self.target_template_args
            )
            parent = self.parent
            # check whether all arguments are really literals
            while parent.parent is not None:
                if isinstance(parent, TemplatableBlock):
                    for arg in target_literal_args:
                        if arg in parent.template_args:
                            return
                parent = parent.parent
            # if so, add this instantiation to appropriate templatable block
            target_block = TemplatableBlock.registry[self.target_name]
            if len(target_block.template_args) != len(target_literal_args):
                raise CompileError(
                    f"Block '{target_block.name}' requires {len(target_block.template_args)} template arguments but {len(target_literal_args)} were given."
                )
            TemplatableBlock.registry[self.target_name].add_instantiation(
                target_literal_args
            )

    def get_literal_args(self, **kwargs) -> tuple[str]:
        return tuple(
            kwargs[template_arg] if template_arg in kwargs else template_arg
            for template_arg in self.target_template_args
        )

    def emit(self, **kwargs) -> str:
        if self.target_name == "return":
            block = None
            parent = self.parent
            while parent.parent is not None:
                if isinstance(parent, Procedure):
                    block = parent
                    break
                parent = parent.parent
            if block is None:
                raise CompileError(
                    "Cannot use the 'return' label outside of a procedure block."
                )
            self.target_template_args = block.template_args
            target = block.get_return_target(self.get_literal_args(**kwargs))
        else:
            block = TemplatableBlock.registry[self.target_name]
            b = block
            while b is not None:
                if isinstance(b, TemplatableBlock):
                    if len(b.template_args) != 0:
                        target = block.get_literal_target(
                        self.get_literal_args(**kwargs), **kwargs)
                        break
                b = b.parent
            if b is None:
                target = self.target_name

        return f"{self.ins} {target}"


class CallInstruction(JumpInstruction):
    def emit(self, **kwargs) -> str:
        proc = TemplatableBlock.registry[self.target_name]
        if not isinstance(proc, Procedure):
            raise CompileError(
                f"Cannot use 'call' on {self.target_name} because it is not a procedure."
            )
        s = ""
        if self.ins == "calltrue":
            s = f"jfalse {proc.get_return_target(self.get_literal_args(**kwargs))}\n"
        elif self.ins == "callfalse":
            s = f"jtrue {proc.get_return_target(self.get_literal_args(**kwargs))}\n"

        s += proc.inline(self.get_literal_args(**kwargs))
        return s


class CompileError(Exception):
    pass


def compile(source: BufferedReader) -> str:
    root = Block()
    root.parse(source)
    print(root)
    root.recurse(JumpInstruction.register_target)
    return root.emit_children(0)


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
    "powerup_nearby"
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
        source = re.sub(
            f"(jmp|jtrue|jfalse) +{k}(\\s)", f"\\1 {v}\\2", source
        )
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
