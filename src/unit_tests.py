import unittest
from data_leakage import (
    count_unmatched_parens,
    dataleakage_class,
    clean_data,
    last_unmatched_opening_pos,
)


class TestParenCounting(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(count_unmatched_parens(""), (0, 0))
        self.assertEqual(count_unmatched_parens("()"), (0, 0))
        self.assertEqual(count_unmatched_parens("("), (1, 0))
        self.assertEqual(count_unmatched_parens(")"), (0, 1))

        self.assertEqual(count_unmatched_parens("foo"), (0, 0))

        self.assertEqual(count_unmatched_parens("()foo"), (0, 0))
        self.assertEqual(count_unmatched_parens("(foo)"), (0, 0))
        self.assertEqual(count_unmatched_parens("foo()"), (0, 0))
        self.assertEqual(count_unmatched_parens("foo(foo)foo"), (0, 0))

        self.assertEqual(count_unmatched_parens("foo("), (1, 0))
        self.assertEqual(count_unmatched_parens("(foo"), (1, 0))

        self.assertEqual(count_unmatched_parens(")foo"), (0, 1))
        self.assertEqual(count_unmatched_parens("foo)"), (0, 1))

    def test_multiple(self):
        self.assertEqual(
            count_unmatched_parens(
                "(everything here is correct)  (((Triple parentheses are an antisemitic symbol)))"
            ),
            (0, 0),
        )
        self.assertEqual(
            count_unmatched_parens(
                "(()()  (((Triple parentheses are an antisemitic symbol))))"
            ),
            (0, 0),
        )
        self.assertEqual(count_unmatched_parens("((()"), (2, 0))
        self.assertEqual(count_unmatched_parens("())))"), (0, 3))
        self.assertEqual(count_unmatched_parens("()) () ("), (1, 1))


class TestDataleakageClass(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(dataleakage_class("", 0.5), "OK")
        self.assertEqual(dataleakage_class("Hello World", 0.5), "OK")
        self.assertEqual(
            dataleakage_class("Hello World (and other planets)", 0.5), "OK"
        )

        self.assertEqual(
            dataleakage_class("Hello World ( sdf sdf sdf sdf sd sd fsd fsd sd ", 0.5),
            "NEG",
        )
        self.assertEqual(dataleakage_class("Hello World (", 0.5), "OK")
        self.assertEqual(dataleakage_class("Hello World )", 0.5), "POS")

        self.assertEqual(
            dataleakage_class("(Hello World) ( s dfdsf sdf sdf sdf sdf sdf sdf ", 0.5),
            "NEG",
        )
        self.assertEqual(dataleakage_class("(Hello World) )", 0.5), "POS")

        self.assertEqual(dataleakage_class(")(", 0.5), "UNKNOWN")
        self.assertEqual(dataleakage_class(")()()(", 0.5), "UNKNOWN")

        self.assertEqual(dataleakage_class(":(", 0.5), "NEG")
        self.assertEqual(dataleakage_class(":)", 0.5), "POS")
        self.assertEqual(dataleakage_class(":( :)", 0.5), "UNKNOWN")

    def test_second_param(self):
        for w in ["", " ", "hello", "hello ()", "(())"]:
            for n in map(lambda x: x / 10, range(-30, 100)):
                self.assertEqual(dataleakage_class(w, n), "OK")

        self.assertEqual(
            dataleakage_class("(Hello World) ( s dfdsf sdf sdf sdf sdf sdf sdf ", 0.1),
            "OK",
        )
        self.assertEqual(
            dataleakage_class("(      ", -1),
            "OK",
        )
        self.assertEqual(
            dataleakage_class("((      ", -1),
            "NEG",
        )
        self.assertEqual(
            dataleakage_class("foo bar b(az", 2 / 3),
            "OK",
        )
        self.assertEqual(
            dataleakage_class("12345(789", 2 / 3),
            "NEG",
        )
        self.assertEqual(
            dataleakage_class("12345(789", 1 / 2),
            "OK",
        )
        self.assertEqual(
            dataleakage_class((" " * 500) + "(", 1),
            "NEG",
        )


class CleanData(unittest.TestCase):
    def test_simple(self):
        n, p = clean_data(
            ["This sucks", "I hate it (", "Everything is awesome )", "Cool ( :)"],
            ["This is awesome", "I love it )", "NOOOO ( NOOOO NOOOO", "Damn :( )"],
        )
        self.assertEqual(n, ["This sucks", "I hate it"])
        self.assertEqual(p, ["This is awesome", "I love it"])


class LastUnmatchedOpeningPos(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(0, last_unmatched_opening_pos("("))
        self.assertEqual(0, last_unmatched_opening_pos("( 1 2 3"))
        self.assertEqual(0, last_unmatched_opening_pos("(() 1 2 3"))
        self.assertEqual(1, last_unmatched_opening_pos("((() 1 2 3"))
        self.assertEqual(1, last_unmatched_opening_pos("((() 1 2 3 ()"))
        self.assertEqual(4, last_unmatched_opening_pos("((()("))


if __name__ == "__main__":
    unittest.main()
