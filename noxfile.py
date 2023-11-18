import nox


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install(".[tests]")
    session.run("pytest", *session.posargs)
