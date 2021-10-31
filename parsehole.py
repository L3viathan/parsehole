import re
import dis
import builtins
import hashlib
from io import BytesIO
from collections import deque

# All Token subclasses:
TOKENS = []
# All Tree subclasses:
TREES = []
# map from Token/Tree subclass name to the class itself:
NAMES = {}
# all precedence levels that are used:
LEVELS = set()


# bytecode instructions (see the __build_class__ shenanigans below):
BUILD_LIST = dis.opmap["BUILD_LIST"]
STORE_NAME = dis.opmap["STORE_NAME"]
LOAD_NAME = dis.opmap["LOAD_NAME"]
LOAD_METHOD = dis.opmap["LOAD_METHOD"]
CALL_METHOD = dis.opmap["CALL_METHOD"]
POP_TOP = dis.opmap["POP_TOP"]
MAKE_FUNCTION = dis.opmap["MAKE_FUNCTION"]


# tiny mixin so we don't have to implement this on Rule, TokenMeta and TreeMeta
class Addable:
    def __add__(self, other):
        if not isinstance(other, Addable):
            return NotImplemented
        if isinstance(self, Rule):
            return Rule(self.parts + [other])
        return Rule([self, other])


# this represents one rule of a tree, e.g. expr -> expr op expr
class Rule(Addable):
    def __init__(self, parts):
        self.parts = parts

    def __repr__(self):
        return "<[" + " + ".join(repr(part) for part in self.parts) + "]>"

    def __len__(self):
        return len(self.parts)

    def __iter__(self):
        yield from iter(self.parts)


# Metaclass for Token; mostly needed for supporting __add__
class TokenMeta(type, Addable):
    def __repr__(self):
        return self.__name__

    @property
    def level(self):
        return self.kwargs.get("level", 0)


# a terminal node
class Token(metaclass=TokenMeta):
    def __init__(self, string):
        self.value = string

    def __init_subclass__(cls, **kwargs):
        cls.kwargs = kwargs
        # matching rule is in docstring:
        cls.regex = re.compile(cls.__doc__)
        TOKENS.append(cls)
        NAMES[cls.__name__] = cls

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value!r})"

    def eval(self):
        return self.value


# to support "recursion" inside a class definition, we put a placeholder into
# the class dict. This placeholder knows the name of the Tree and will
# eventually be replaced by the real class.
class Placeholder(Addable):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    # not really needed anymore, but just so you could theoretically
    # instantiate it as well
    def __call__(self, *args, **kwargs):
        return NAMES[self.name](*args, **kwargs)

    def replace(self):
        return NAMES[self.name]


# metaclass for Tree. Needed for __add__ support, and "recursion"
class TreeMeta(type, Addable):
    def __prepare__(name, bases, **kwargs):
        prepared = object.__prepare__(name, bases)
        # we insert a placeholder object into the class namespace under the
        # name of the class. That way, when executing the class body, we don't
        # get a NameError when using "recursive" rules (such as expr -> op
        # expr).
        prepared[name] = Placeholder(name)
        return prepared

    @property
    def level(self):
        return self.kwargs.get("level", 0)

    def __repr__(self):
        return self.__name__


class Tree(metaclass=TreeMeta):
    def __init__(self, parts):
        self.parts = parts

    def __init_subclass__(cls, **kwargs):
        cls.kwargs = kwargs
        LEVELS.add(cls.level)
        TREES.append(cls)
        NAMES[cls.__name__] = cls
        # normalization: if a rule outputs only a single Tree/Token, it will
        # not be a Rule object yet, we fix that here
        cls.rules = [
            rule if isinstance(rule, Rule) else Rule([rule]) for rule in cls.rules
        ]
        # if we have any recursive rules, we replace the placeholders with
        # actual references to the class
        for rule in cls.rules:
            rule.parts = [
                part.replace() if isinstance(part, Placeholder) else part
                for part in rule.parts
            ]

    # like the function parse, but enforcing that the resulting Tree is _this_
    # kind of tree
    @classmethod
    def parse(cls, string):
        tokens = list(tokenize(string))
        tree = parse(tokens, start=cls)
        if not isinstance(tree, cls):
            raise ValueError(f"Parsed tree was {type(tree)}, not {cls.__name__}")
        return tree

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({', '.join(repr(part) for part in self.parts)})"
        )


def tokenize(string):
    """Turn a string into a sequence of tokens."""
    while string:
        # the sorting enforces the precedence levels for matching
        for token in sorted(TOKENS, key=lambda t: t.level):
            if match := token.regex.match(string):
                # Tokens with ignore=True (could be e.g. Whitespace) are not
                # emitted
                if not token.kwargs.get("ignore", False):
                    yield token(match.string[: match.end()])
                string = string[match.end() :]
                break
        else:
            raise ValueError(f"Can't tokenize remaining value: {string!r}")


def reduce(stack, rule, tree):
    """The reduce step of our shift-reduce parser"""
    rule_len = len(rule)
    for item, rule_part in zip(stack[-rule_len:], rule):
        if not isinstance(item, rule_part):
            return False
    parts = [stack.pop() for _ in range(rule_len)]
    stack.append(tree(parts))
    return True


# this is basically just an iterator that goes over a (sorted) sequence of
# ints, except it can be reset to the beginning at any time. Used to enforce
# precedence for rules.
class LevelMaster:
    def __init__(self, levels):
        self.levels = sorted(levels, reverse=True)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            rval = self.levels[self.index]
            self.index += 1
            return rval
        except IndexError:
            raise StopIteration from None

    def reset(self):
        self.index = 0


# so here's the thing: I've done this some years ago at Uni, but I was to lazy
# to look it up. This might not always work, might be inefficient, etc. It's
# anyways kind of the boring part of this lib (nothing special here, except for
# its badness)
def parse(sequence, start=None):
    if len(sequence) == 1:
        return sequence[0]
    agenda = deque(sequence)
    stack = [agenda.popleft()]
    level_master = LevelMaster(LEVELS)
    while len(stack) != 1 or agenda:
        for level in level_master:
            changed = True
            while changed:
                changed = False
                # reduce
                for tree in TREES:
                    if tree.level != level:
                        continue
                    for rule in tree.rules:
                        if len(rule) > len(stack):
                            continue
                        if reduce(stack, rule, tree):
                            level_master.reset()
                            break
                # shift
                if agenda:
                    stack.append(agenda.popleft())
                    changed = True
        if len(stack) > 1 and not agenda:
            agenda.extend(stack.pop() for _ in range(len(stack)))
        level_master.reset()
    return stack[0]


old_build_class = builtins.__build_class__


# now this _is_ fun: because there's no hook in metaclasses to customize how
# the class body of their instances is called, we have to hook into the global
# hook of how class definition statements are executed. We bail early if the
# class that's being defined isn't ours, of course, but still, this seems kind
# of messy. I'd love a __build_class__ method on the metaclass as well..
# Anyways, the purpose of this is to transform the code from this:
# class Foo(Tree):
#     X
#     Y + Z + Foo
# into this:
# class Foo(Tree):
#     rules = []
#     rules.append(X)
#     rules.append(Y + Z + Foo)
# We do this by adding the names "rules" and "append" to co_names, and then
# patching the bytecode to wrap these empty, discarded statements that would
# normally be garbage-collected into rules.append()-calls, so they can be
# accessed later.
# Oh, and I hope I understood co_stacksize correctly, if not, your program
# might just crash.
def __build_class__(func, name, *bases, **kwargs):
    if Tree in bases:
        code = func.__code__
        # append "append" and "rules" to co_names
        co_names = tuple([*code.co_names, "append", "rules"])
        i_append = len(co_names) - 2
        i_rules = len(co_names) - 1
        # set co_stacksize to at least 4
        co_stacksize = min(code.co_stacksize, 4)
        # go through the bytecode:
        co_code = BytesIO()
        # first four opcodes are writing __qualname__ etc.; don't touch
        co_code.write(code.co_code[:8])
        # in the beginning, add BUILD_LIST 0, STORE_NAME rules
        co_code.write(bytes([BUILD_LIST, 0, STORE_NAME, i_rules]))
        buffer = []
        reached_makefunction = False
        for i in range(len(code.co_code) // 2):
            instruction, argument = code.co_code[i * 2 : i * 2 + 2]
            if i in range(4):  # first four opcodes are doing class stuff, see above
                continue
            # find groups of things until a POP_TOP, before the first MAKE_FUNCTION:
            if not reached_makefunction and instruction == POP_TOP:
                # replace them with LOAD_NAME rules, LOAD_METHOD append, ...,
                # CALL_METHOD 1, POP_TOP
                co_code.write(bytes([LOAD_NAME, i_rules, LOAD_METHOD, i_append]))
                co_code.write(bytes(buffer))
                co_code.write(bytes([CALL_METHOD, 1, POP_TOP, 0]))
                buffer = []
                continue
            buffer.extend([instruction, argument])
            if instruction == MAKE_FUNCTION:
                # we've reached the first method definition, no more
                # modifications necessary
                reached_makefunction = True

        co_code.write(bytes(buffer))
        co_code = co_code.getvalue()

        # the last time I did something like this, CodeType.replace didn't
        # exist yet. Hooray!
        func.__code__ = code.replace(
            co_names=co_names,
            co_stacksize=co_stacksize,
            co_code=co_code,
        )
    return old_build_class(func, name, *bases, **kwargs)


builtins.__build_class__ = __build_class__
