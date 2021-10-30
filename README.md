# parsehole

```python
from parsehole import Token, Tree, parse
```

# Tokens

Define tokens like this:

```python
class Number(Token):
    r"\d+"
```

Tokens can be set to be ignored, and be assigned precedence levels (for
ambiguous tokenization rules):

```python
class Whitespace(Token, ignore=True, level=-1):
    r"\s+"
```

They can also have a configured value, for example for the `Number` token above:

```python
@property
def value(self):
    return int(self.string)
```


# Trees

Non-terminals are called trees. They have a `rules` attribute that contains all
possible expansions of this tree type. Note that recursive definitions "just
work" (in the sense that there are no NameErrors occurring despite the class
not being defined yet).

```python
class AddExpression(Tree):
    rules = (
        AddExpression + AddOperator + AddExpression
        | MulExpression
    )
```

# Parsing

Call `parse()` on a string to get a parse tree (if possible). Every tree also
has a class method of the same name that raises an error if the resulting tree
isn't of the specified type.

# Evaluation

Both trees and tokens can have a `eval` method which defines how to get to the
tree/token's value.
