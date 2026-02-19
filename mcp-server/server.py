from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math")


@mcp.tool()
def multiply_subtool(a: float, b: float) -> float:
    """
    Multiplica dois números.

    Args:
        a: O primeiro número.
        b: O segundo número.

    Returns:
        O resultado da multiplicação de a e b.
    """

    return a * b


@mcp.tool()
def add_subtool(a: float, b: float) -> float:
    """
    Soma dois números.

    Args:
        a: O primeiro número.
        b: O segundo número.

    Returns:
        O resultado da soma de a e b.
    """

    return a + b


@mcp.tool()
def subtract_subtool(a: float, b: float) -> float:
    """
    Subtrai dois números.

    Args:
        a: O primeiro número.
        b: O segundo número.

    Returns:
        O resultado da subtração de a e b.
    """

    return a - b


@mcp.tool()
def divide_subtool(a: float, b: float) -> float:
    """
    Divide dois números.

    Args:
        a: O primeiro número (dividendo).
        b: O segundo número (divisor).

    Returns:
        O resultado da divisão de a por b, ou uma mensagem de erro se b for zero.
    """

    if b == 0:
        return "Divisão por zero não é permitida."

    return a / b


if __name__ == "__main__":
    mcp.run(transport="stdio")