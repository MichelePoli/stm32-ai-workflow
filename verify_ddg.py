
import sys
try:
    from agno.tools.duckduckgo import DuckDuckGoTools
    print("Imported from agno.tools.duckduckgo")
except ImportError as e1:
    print(f"Failed agno.tools.duckduckgo: {e1}")
    try:
        from agno.tools.duckduckgo_search import DuckDuckGoTools
        print("Imported from agno.tools.duckduckgo_search")
    except ImportError as e2:
        print(f"Failed agno.tools.duckduckgo_search: {e2}")
        sys.exit(1)
