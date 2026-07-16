from book_list import (
    add_book,
    change_status,
    find_books,
    main,
    search_books,
    show_books,
)


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


def test_add_book_reasks_for_empty_name(monkeypatch, capsys):
    books = []
    answers = iter(["", "小王子", "圣埃克苏佩里", "童话", "已读"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    add_book(books)

    assert books[0]["书名"] == "小王子"
    assert "书名不能为空" in capsys.readouterr().out


def test_add_book_ignores_existing_book(monkeypatch, capsys):
    books = [{"书名": "活着", "作者": "余华", "类型": "文学", "状态": "已读"}]
    answers = iter(["活着", "余华"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    add_book(books)

    assert len(books) == 1
    assert "该书已在书单中" in capsys.readouterr().out


def test_search_books_prints_no_result(monkeypatch, capsys):
    books = [{"书名": "三体", "作者": "刘慈欣", "类型": "科幻", "状态": "在读"}]
    monkeypatch.setattr("builtins.input", lambda _: "历史")

    search_books(books)

    assert "没有找到相关书籍" in capsys.readouterr().out


def test_change_status(monkeypatch, capsys):
    books = [
        {"书名": "活着", "作者": "余华", "类型": "文学", "状态": "已读"},
        {"书名": "三体", "作者": "刘慈欣", "类型": "科幻", "状态": "在读"},
    ]
    answers = iter(["2", "3"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    change_status(books)

    assert books[1]["状态"] == "已读"
    assert "《三体》的阅读状态已修改为：已读" in capsys.readouterr().out


def test_change_status_rejects_bad_book_number(monkeypatch, capsys):
    books = [{"书名": "活着", "作者": "余华", "类型": "文学", "状态": "已读"}]
    monkeypatch.setattr("builtins.input", lambda _: "9")

    change_status(books)

    assert books[0]["状态"] == "已读"
    assert "书籍编号无效" in capsys.readouterr().out


def test_change_status_rejects_bad_status(monkeypatch, capsys):
    books = [{"书名": "活着", "作者": "余华", "类型": "文学", "状态": "已读"}]
    answers = iter(["1", "4"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    change_status(books)

    assert books[0]["状态"] == "已读"
    assert "状态编号无效" in capsys.readouterr().out


def test_main_can_view_and_exit(monkeypatch, capsys):
    answers = iter(["2", "5"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    main()

    out = capsys.readouterr().out
    assert "我的书单" in out
    assert "修改阅读状态" in out
    assert "《活着》" in out
    assert "程序已退出" in out
