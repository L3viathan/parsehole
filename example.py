from parsehole import Token, Rule

class Number(Token):
    r"\d+"

    @property
    def value(self):
        return int(self.string)

class Whitespace(Token, ignore=True, level=-1):
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


class MulExpression(Rule, level=1):
    rule = (
        (MulExpression + MulOperator + MulExpression)
        | Number
    )
    def eval(self):
        match self.parts:
            case (MulExpression() as e1, Operator() as op, MulExpression() as e2):
                return op.eval(e1.eval(), e2.eval())
            case (_ as val,):
                return val.eval()


class AddExpression(Rule):
    rule = (
        (AddExpression + AddOperator + AddExpression)
        | MulExpression
    )
    def eval(self):
        match self.parts:
            case (AddExpression() as e1, Operator() as op, AddExpression() as e2):
                return op.eval(e1.eval(), e2.eval())
            case (_ as val,):
                return val.eval()



print(AddExpression.rule)
expr = AddExpression.parse("2 * 3 + 4")
print(expr.eval(), "should be 10")
