# Book List Program Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small Chinese command-line book list program with add, view, keyword search, and exit functions.

**Architecture:** A single `book_list.py` file owns the sample list and four small functions. Pytest captures standard input and output so the behavior is verified without adding testing helpers to the production code.

**Tech Stack:** Python 3.10+, pytest

---

### Task 1: Add, display, and search behavior

**Files:**
- Create: `book_list.py`
- Create: `tests/test_book_list.py`

- [ ] **Step 1: Write the failing tests**

```python
from book_list import add_book, find_books, show_books


def test_add_book(monkeypatch):
    books = []
    answers = iter(["小王子", "圣埃克苏佩里", "童话", "已读"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    add_book(books)

    assert books == [{
        "书名": "小王子",
        "作者": "圣埃克苏佩里",
        "类型": "童话",
        "状态": "已读",
    }]


def test_show_books(capsys):
    books = [{"书名": "活着", "作者": "余华", "类型": "文学", "状态": "已读"}]

    show_books(books)

    out = capsys.readouterr().out
    assert "1.《活着》" in out
    assert "余华" in out
    assert "文学" in out
    assert "已读" in out


def test_find_books_matches_every_field():
    books = [
        {"书名": "三体", "作者": "刘慈欣", "类型": "科幻", "状态": "在读"},
        {"书名": "活着", "作者": "余华", "类型": "文学", "状态": "已读"},
    ]

    assert find_books(books, "刘慈欣") == [books[0]]
    assert find_books(books, "文学") == [books[1]]
    assert find_books(books, "未读") == []
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_book_list.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'book_list'`.

- [ ] **Step 3: Write the minimal implementation**

```python
def add_book(books):
    name = input("请输入书名：").strip()
    author = input("请输入作者：").strip()
    kind = input("请输入类型：").strip()
    status = input("请输入阅读状态：").strip()
    books.append({"书名": name, "作者": author, "类型": kind, "状态": status})
    print("添加成功！")


def show_books(books):
    if not books:
        print("书单为空。")
        return
    for i, book in enumerate(books, 1):
        print(
            f"{i}.《{book['书名']}》 "
            f"作者：{book['作者']}  类型：{book['类型']}  状态：{book['状态']}"
        )


def find_books(books, key):
    key = key.lower()
    return [
        book for book in books
        if any(key in value.lower() for value in book.values())
    ]
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `python -m pytest tests/test_book_list.py -q`

Expected: `3 passed`.

### Task 2: Input validation and query messages

**Files:**
- Modify: `book_list.py`
- Modify: `tests/test_book_list.py`

- [ ] **Step 1: Add failing validation and query tests**

```python
from book_list import add_book, find_books, search_books, show_books


def test_add_book_reasks_for_empty_name(monkeypatch, capsys):
    books = []
    answers = iter(["", "小王子", "圣埃克苏佩里", "童话", "已读"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    add_book(books)

    assert books[0]["书名"] == "小王子"
    assert "书名不能为空" in capsys.readouterr().out


def test_search_books_prints_no_result(monkeypatch, capsys):
    books = [{"书名": "三体", "作者": "刘慈欣", "类型": "科幻", "状态": "在读"}]
    monkeypatch.setattr("builtins.input", lambda _: "历史")

    search_books(books)

    assert "没有找到相关书籍" in capsys.readouterr().out
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run: `python -m pytest tests/test_book_list.py -q`

Expected: import fails because `search_books` does not exist.

- [ ] **Step 3: Add validation and interactive search**

```python
def get_text(prompt, msg):
    while True:
        text = input(prompt).strip()
        if text:
            return text
        print(msg)


def add_book(books):
    name = get_text("请输入书名：", "书名不能为空。")
    author = get_text("请输入作者：", "作者不能为空。")
    kind = input("请输入类型：").strip()
    status = input("请输入阅读状态：").strip()
    books.append({"书名": name, "作者": author, "类型": kind, "状态": status})
    print("添加成功！")


def search_books(books):
    key = get_text("请输入关键词：", "关键词不能为空。")
    result = find_books(books, key)
    if result:
        show_books(result)
    else:
        print("没有找到相关书籍。")
```

- [ ] **Step 4: Run all book-list tests**

Run: `python -m pytest tests/test_book_list.py -q`

Expected: `5 passed`.

### Task 3: Menu loop and smoke verification

**Files:**
- Modify: `book_list.py`
- Modify: `tests/test_book_list.py`

- [ ] **Step 1: Add a failing menu test**

```python
from book_list import add_book, find_books, main, search_books, show_books


def test_main_can_view_and_exit(monkeypatch, capsys):
    answers = iter(["2", "4"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    main()

    out = capsys.readouterr().out
    assert "我的书单" in out
    assert "《活着》" in out
    assert "程序已退出" in out
```

- [ ] **Step 2: Run the menu test and verify it fails**

Run: `python -m pytest tests/test_book_list.py::test_main_can_view_and_exit -q`

Expected: import fails because `main` does not exist.

- [ ] **Step 3: Add sample books and the menu loop**

```python
books = [
    {"书名": "活着", "作者": "余华", "类型": "文学", "状态": "已读"},
    {"书名": "三体", "作者": "刘慈欣", "类型": "科幻", "状态": "在读"},
    {"书名": "百年孤独", "作者": "加西亚·马尔克斯", "类型": "文学", "状态": "未读"},
]


def main():
    data = books.copy()
    while True:
        print("\n===== 我的书单 =====")
        print("1. 添加书籍")
        print("2. 查看全部")
        print("3. 关键词查询")
        print("4. 退出程序")
        choice = input("请选择：").strip()

        if choice == "1":
            add_book(data)
        elif choice == "2":
            show_books(data)
        elif choice == "3":
            search_books(data)
        elif choice == "4":
            print("程序已退出。")
            break
        else:
            print("输入有误，请重新选择。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/test_book_list.py -q`

Expected: `6 passed`.

- [ ] **Step 5: Run a real command-line smoke test**

Run: `@('2', '3', '科幻', '4') | python book_list.py`

Expected: the output shows all three books, finds 《三体》, and exits normally.
