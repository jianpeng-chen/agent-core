import os
import subprocess
import sys

import pytest

from core.cli import main


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("OPENAI_", "AGENT_")):
            monkeypatch.delenv(name)


def test_demo_once_runs_the_real_tool_loop(capsys):
    assert main(["--demo", "--once", "你好"]) == 0
    assert "工具返回：你好" in capsys.readouterr().out


def test_repl_retains_session_and_handles_blank_input_and_exit(monkeypatch, capsys):
    inputs = iter([" ", "first", "second", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "工具返回：first" in out
    assert "工具返回：second" in out


@pytest.mark.parametrize("exception,code", [(EOFError, 0), (KeyboardInterrupt, 130)])
def test_terminal_exit(monkeypatch, exception, code):
    def read(_):
        raise exception

    monkeypatch.setattr("builtins.input", read)
    assert main(["--demo"]) == code


def test_missing_key_is_a_configuration_error(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--once", "hi"])
    assert error.value.code == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option,value",
    [
        ("--max-steps", "-1"),
        ("--context-turns", "0"),
        ("--temperature", "nan"),
        ("--timeout", "0"),
        ("--timeout", "inf"),
    ],
)
def test_cli_validates_configuration(option, value):
    with pytest.raises(SystemExit) as error:
        main(["--demo", option, value])
    assert error.value.code == 2


def test_invalid_environment_and_cli_override(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "invalid")
    with pytest.raises(SystemExit):
        main(["--demo", "--once", "hi"])
    assert main(["--demo", "--max-steps", "1", "--once", "hi"]) == 0


def test_module_entrypoint_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "core", "--demo", "--once", "smoke"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
    assert "工具返回：smoke" in result.stdout


@pytest.mark.parametrize("from_environment", [False, True])
def test_invalid_url_is_configuration_error(monkeypatch, capsys, from_environment):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    arguments = ["--once", "hello"]
    if from_environment:
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:abc")
    else:
        arguments.extend(["--base-url", "http://localhost:abc"])
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "base_url" in stderr
    assert "Traceback" not in stderr
