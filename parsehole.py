import re
from collections import deque

TOKENS = []
RULES = []
NAMES = {}
LEVELS = set()


class Addable:
    def __add__(self, other):
        if not isinstance(other, Addable):
            return NotImplemented
        if isinstance(self, RuleSequence):
            return RuleSequence(self.parts + [other])
        return RuleSequence([self, other])

    def __or__(self, other):
        if not isinstance(other, Addable):
            return NotImplemented
        if not isinstance(other, RuleSequence):
            other = RuleSequence([other])
        if isinstance(self, RuleOption):
            return RuleOption(self.parts + [other])
        return RuleOption([self, other])


class RuleSequence(Addable):
    def __init__(self, parts):
        self.parts = parts

    def __repr__(self):
        return "<[" + " + ".join(repr(part) for part in self.parts) + "]>"


class RuleOption(Addable):
    def __init__(self, parts):
        self.parts = parts

    def __repr__(self):
        return "<{" + " | ".join(repr(part) for part in self.parts) + "}>"


class TokenMeta(type, Addable):
    def __repr__(self):
        return self.__name__  # TODO: getitem?


class Token(metaclass=TokenMeta):
    def __init__(self, string):
        self.string = string

    @property
    def value(self):
        return self.string

    def __init_subclass__(cls, **kwargs):
        for key, value in kwargs.items():
            setattr(cls, key, value)
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


class RuleMeta(type, Addable):
    def __prepare__(name, bases, **kwargs):
        prepared = object.__prepare__(name, bases)
        prepared[name] = Placeholder(name)  # some kind of placeholder
        return prepared

    @property
    def level(self):
        return self.kwargs.get("level", 0)

    def __repr__(self):
        return self.__name__


class Rule(metaclass=RuleMeta):
    def __init__(self, parts):
        self.parts = parts

    def __init_subclass__(cls, **kwargs):
        cls.kwargs = kwargs
        # normalization: every rule is a option of a sequence
        if isinstance(cls.rule, Token):
            cls.rule = RuleSequence([cls.rule])
        if isinstance(cls.rule, RuleSequence):
            cls.rule = RuleOption([cls.rule])
        LEVELS.add(cls.level)
        RULES.append(cls)
        NAMES[cls.__name__] = cls
        for option in cls.rule.parts:
            option.parts = [
                part.replace() if isinstance(part, Placeholder) else part
                for part in option.parts
            ]

    @classmethod
    def parse(cls, string):
        tokens = list(tokenize(string))
        print("Tokens:", tokens)
        return parse(tokens, start=cls)

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(repr(part) for part in self.parts)})"


def tokenize(string):
    while string:
        for token in TOKENS:
            if match := token.regex.match(string):
                if not getattr(token, "ignore", False):
                    yield token(match.string[: match.end()])
                string = string[match.end() :]
                break
        else:
            raise ValueError(f"Can't tokenize remaining value: {string!r}")


class ParseError(Exception):
    pass


def reduce(stack, rule_choice, rule):
    rule_len = len(rule_choice.parts)
    for item, rule_part in zip(stack[-rule_len:], rule_choice.parts):
        if not isinstance(item, rule_part):
            return False
    parts = [stack.pop() for _ in range(rule_len)]
    stack.append(rule(parts))
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
                print("main loop:", stack, agenda)
                # reduce
                for rule in RULES:
                    if rule.level != level:
                        continue
                    for rule_choice in rule.rule.parts:
                        if len(rule_choice.parts) > len(stack):
                            continue
                        if reduce(stack, rule_choice, rule):
                            level_master.reset()
                            changed = True
                if changed:
                    break
                if agenda:
                    stack.append(agenda.popleft())
                    changed = True
        if len(stack) > 1 and not agenda:
            agenda.extend(stack.pop() for _ in range(len(stack)))
        level_master.reset()
    print("after loop:", stack, agenda)
    return stack[0]
