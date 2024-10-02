# Smollbonk : Bonkbly Preprocessor

Smollbonk is both a simple macro language for the [Bonkbly](https://botbattle.be/bonkbly) assembly language and its preprocessor/compiler.

It comes with the following major features:
- Full Bonkly parsing: detects if source code contains invalid Bonkly.
- Procedures (keywords `proc` and `call`) that allow for always-inlined function-like control flow.
- Templating allowing for efficient code generation, facilitating more compact source code (keyword `template`).
- Repeat blocks enabling duplication of instruction groups.
- Optional minification (register renaming, label renaming and whitespace stripping) allowing great reductions in output code size

Example Smollbonk bots and their Bonkly output are available under the `bots` directory.

## Compiling

The Smollbonk Python compiler has only the `click` package as a dependency. 
If you happen to have [Poetry](https://python-poetry.org/docs/#installation) already installed:
```sh
# Creates venv, installs dependencies and activates venv
poetry install --no-root
```

Otherwise you may install Poetry or simply use Python's builtin `venv` module:
<table>
    <tr><td>Unix/macOS</td><td>Windows</td></tr>
    <tr>
        <td>

```sh
# Create venv
python -m venv .venv 
# Activate venv
source .venv/bin/activate 
# Install dependencies
python3 -m pip install -r requirements.txt 
```
</td>
<td>

```sh
# Create venv
py -m venv .venv 
# Activate venv
.venv\Scripts\activate  
# Install dependencies
py -m pip install -r requirements.txt 
```

</td>
</tr>
</table>



Smollbonk source files may then be compiled with `smollbonk.py`:
```sh
# Compiles the bot.sbonk Smollbonk file to Bonkly in bot.bonk
python smollbonk.py compile bot.sbonk bot.bonk
# Previous, but also minifies the result
python smollbonk.py compile -m bot.sbonk bot.bonk
# Minifies a standalone Bonkly file 
python smollbonk.py minify bot.bonk bot-min.bonk
```

## Parsing

The Smollbonk compiler includes a recursive parser for the Bonkly language except for instruction arguments.

What the parser cannot understand will not be emitted in the final compiled Bonkly code: this includes comments but also mistyped instructions or other invalid syntactical constructs.

When parsing, the entire Abstract Syntax Tree is printed to `stdout`: this may help in the diagnosis of parsing errors. Unparsable lines are also displayed to the user.

## Blocks

In Smollbonk, labels, procedures and their templated variants all define "blocks" of instructions which, in some contexts, will have semantic meaning. 
Blocks are defined by a block title followed by a colon, after which the block's body may begin (but may be empty!).

Smollbonk uses indentation (4 spaces) in order to figure out which instructions belong to a block.
A single instruction may also trail after the block's colon (on the same line): Smollbonk will count this instruction as inside the block.

Block names may not contain the sequence "__" (double underscore), as this could interfere with the templating engine.

### Procedures

### Templating

#### Template Specialisation block

### Repeat




## Minification

Smollbonk's minification procedure is entirely separate from compilation, and thus can be used completely independently of it.

