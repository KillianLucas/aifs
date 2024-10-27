from aifs import search
import os

def setup_environment():
    os.environ['AIFS_MINIMAL_PYTHON_INDEXING'] = 'false'

def run_search_test(query, path):
    results = search(query, path=path)
    print(results)
    assert results

def test_search_this():
    setup_environment()
    current_dir_path = os.path.dirname(os.path.realpath(__file__))
    run_search_test("test search", current_dir_path)

def test_search_desktop():
    setup_environment()
    desktop_path = os.path.expanduser("~/Desktop")
    run_search_test("forest gump", desktop_path)
