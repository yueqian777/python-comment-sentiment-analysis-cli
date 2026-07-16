books = [
    {"书名": "活着", "作者": "余华", "类型": "文学", "状态": "已读"},
    {"书名": "三体", "作者": "刘慈欣", "类型": "科幻", "状态": "在读"},
    {"书名": "百年孤独", "作者": "加西亚·马尔克斯", "类型": "文学", "状态": "未读"},
]


def get_text(prompt, msg):
    while True:
        text = input(prompt).strip()
        if text:
            return text
        print(msg)


def add_book(books):
    name = get_text("请输入书名：", "书名不能为空。")
    author = get_text("请输入作者：", "作者不能为空。")

    for book in books:
        same_name = book["书名"].lower() == name.lower()
        same_author = book["作者"].lower() == author.lower()
        if same_name and same_author:
            print("该书已在书单中，不再重复添加。")
            return

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


def search_books(books):
    key = get_text("请输入关键词：", "关键词不能为空。")
    result = find_books(books, key)
    if result:
        show_books(result)
    else:
        print("没有找到相关书籍。")


def change_status(books):
    if not books:
        print("书单为空。")
        return

    show_books(books)
    num = input("请输入书籍编号：").strip()
    if not num.isdigit() or not 1 <= int(num) <= len(books):
        print("书籍编号无效。")
        return

    print("1. 未读")
    print("2. 在读")
    print("3. 已读")
    choice = input("请选择新的阅读状态：").strip()
    states = {"1": "未读", "2": "在读", "3": "已读"}
    if choice not in states:
        print("状态编号无效。")
        return

    book = books[int(num) - 1]
    book["状态"] = states[choice]
    print(f"《{book['书名']}》的阅读状态已修改为：{book['状态']}")


def main():
    data = books.copy()

    while True:
        print("\n===== 我的书单 =====")
        print("1. 添加书籍")
        print("2. 查看全部")
        print("3. 关键词查询")
        print("4. 修改阅读状态")
        print("5. 退出程序")
        choice = input("请选择：").strip()

        if choice == "1":
            add_book(data)
        elif choice == "2":
            show_books(data)
        elif choice == "3":
            search_books(data)
        elif choice == "4":
            change_status(data)
        elif choice == "5":
            print("程序已退出。")
            break
        else:
            print("输入有误，请重新选择。")


if __name__ == "__main__":
    main()
