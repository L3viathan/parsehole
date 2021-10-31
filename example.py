from parsehole import Token, Tree


class Number(Token):
    r"\d+"

    def eval(self):
        return int(self.value)


class Whitespace(Token, ignore=True):
    r"\s+"


class AddOperator(Token):
    r"[+-]"

    def eval(self, val1, val2):
        match self.value:
            case "+":
                return val1 + val2
            case "-":
                return val1 - val2


class MulOperator(Token):
    r"[*/]"

    def eval(self, val1, val2):
        match self.value:
            case "*":
                return val1 * val2
            case "/":
                return val1 / val2


class MulExpression(Tree, level=1):
    MulExpression + MulOperator + MulExpression
    Number

    def eval(self):
        match self.parts:
            case (MulExpression() as e1, MulOperator() as op, MulExpression() as e2):
                return op.eval(e1.eval(), e2.eval())
            case (_ as val,):
                return val.eval()


class AddExpression(Tree):
    AddExpression + AddOperator + AddExpression
    MulExpression

    def eval(self):
        match self.parts:
            case (AddExpression() as e1, AddOperator() as op, AddExpression() as e2):
                return op.eval(e1.eval(), e2.eval())
            case (_ as val,):
                return val.eval()


assert AddExpression.parse("2 * 3 + 4").eval() == 10
assert AddExpression.parse("2 + 3 * 4").eval() == 14
