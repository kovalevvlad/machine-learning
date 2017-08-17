def pytest_generate_tests(metafunc):
    parallelism = 'disable_parallelism'
    if parallelism in metafunc.fixturenames:
        metafunc.parametrize(parallelism, [True, False])