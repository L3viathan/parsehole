import re
from collections import deque

TOKENS = []
TREES = []
NAMES = {}
LEVELS = set()


class Addable:
    def __add__(self, other):
        if not isinstance(other, Addable):
            return NotImplemented
        if isinstance(self, Rule):
            return Rule(self.parts + [other])
        return Rule([self, other])

    def __or__(self, other):
        if not isinstance(other, Addable):
            return NotImplemented
        if not isinstance(other, Rule):
            other = Rule([other])
        if isinstance(self, TreeOptionPlaceholder):
            return TreeOptionPlaceholder(self.parts + [other])
        return TreeOptionPlaceholder([self, other])


class Rule(Addable):
    def __init__(self, parts):
        self.parts = parts

    def __repr__(self):
        return "<[" + " + ".join(repr(part) for part in self.parts) + "]>"


class TreeOptionPlaceholder(Addable):
    def __init__(self, parts):
        self.parts = parts

    def __repr__(self):
        return "<{" + " | ".join(repr(part) for part in self.parts) + "}>"


class TokenMeta(type, Addable):
    def __repr__(self):
        return self.__name__  # TODO: getitem?

    @property
    def level(self):
        return self.kwargs.get("level", 0)


class Token(metaclass=TokenMeta):
    def __init__(self, string):
        self.string = string

    @property
    def value(self):
        return self.string

    def __init_subclass__(cls, **kwargs):
        cls.kwargs = kwargs
        cls.regex = re.compile(cls.__doc__)
        TOKENS.append(cls)
        NAMES[cls.__name__] = cls

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value!r})"

    def eval(self):
        return self.value


class Placeholder(Addable):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        return NAMES[self.name](*args, **kwargs)

    def replace(self):
        return NAMES[self.name]


class TreeMeta(type, Addable):
    def __prepare__(name, bases, **kwargs):
        prepared = object.__prepare__(name, bases)
        prepared[name] = Placeholder(name)  # some kind of placeholder
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
        # normalization: every rules is a option of a sequence
        if isinstance(cls.rules, Token):
            cls.rules = Rule([cls.rules])
        if isinstance(cls.rules, Rule):
            cls.rules = TreeOptionPlaceholder([cls.rules])
        cls.rules = cls.rules.parts
        LEVELS.add(cls.level)
        TREES.append(cls)
        NAMES[cls.__name__] = cls
        for option in cls.rules:
            option.parts = [
                part.replace() if isinstance(part, Placeholder) else part
                for part in option.parts
            ]

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
    while string:
        for token in sorted(TOKENS, key=lambda t: t.level):
            if match := token.regex.match(string):
                if not token.kwargs.get("ignore", False):
                    yield token(match.string[: match.end()])
                string = string[match.end() :]
                break
        else:
            raise ValueError(f"Can't tokenize remaining value: {string!r}")


class ParseError(Exception):
    pass


def reduce(stack, rule, tree):
    rule_len = len(rule.parts)
    for item, rule_part in zip(stack[-rule_len:], rule.parts):
        if not isinstance(item, rule_part):
            return False
    parts = [stack.pop() for _ in range(rule_len)]
    stack.append(tree(parts))
    return True


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
                        if len(rule.parts) > len(stack):
                            continue
                        if reduce(stack, rule, tree):
                            level_master.reset()
                            break
                if agenda:
                    stack.append(agenda.popleft())
                    changed = True
        if len(stack) > 1 and not agenda:
            agenda.extend(stack.pop() for _ in range(len(stack)))
        level_master.reset()
    return stack[0]
