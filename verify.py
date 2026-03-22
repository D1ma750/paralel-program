import numpy as np
import os


def load_matrix(file_path):
    """Чтение квадратной матрицы из файла."""
    try:
        with open(file_path, "r") as f:
            n = int(f.readline().strip())
            data = []

            for i in range(n):
                row = list(map(float, f.readline().split()))
                if len(row) != n:
                    print(f"Ошибка: строка {i+1} файла {file_path} имеет неверную длину")
                    return None
                data.append(row)

        return np.array(data)

    except FileNotFoundError:
        print(f"Файл {file_path} отсутствует")
        return None

    except Exception as e:
        print(f"Ошибка чтения {file_path}: {e}")
        return None


def check_result(size):
    """Проверка результата умножения для одного размера."""

    print(f"\nПроверка матриц {size}x{size}")

    fileA = f"matrix_a_{size}.txt"
    fileB = f"matrix_b_{size}.txt"
    fileC = f"matrix_c_{size}.txt"

    if not (os.path.exists(fileA) and os.path.exists(fileB) and os.path.exists(fileC)):
        print("Не найдены необходимые файлы.")
        return None

    A = load_matrix(fileA)
    B = load_matrix(fileB)
    C_prog = load_matrix(fileC)

    if A is None or B is None or C_prog is None:
        return None

    print("Вычисление эталонного результата (NumPy)...")

    C_ref = np.matmul(A, B)

    ok = np.allclose(C_ref, C_prog, atol=1e-6)

    if ok:
        print("Результат корректен")
    else:
        print("Обнаружено несоответствие")

    return {
        "size": size,
        "ok": ok,
        "files": (fileA, fileB, fileC)
    }


def print_summary(results):
    """Вывод итоговой таблицы."""

    print("\n" + "-" * 65)
    print("| Размер | Статус    | Макс. ошибка | Средняя ошибка |")
    print("-" * 65)

    for r in results:
        status = "OK" if r["ok"] else "ERROR"

        print(f"| {r['size']:6d} | {status:9s} |")

    print("-" * 65)


def save_report(results):
    """Сохранение отчета проверки."""

    with open("verification_reports.txt", "w", encoding="UTF-8") as f:

        f.write("ОТЧЕТ ПРОВЕРКИ УМНОЖЕНИЯ МАТРИЦ\n")
        f.write("=" * 50 + "\n\n")

        for r in results:

            status = "УСПЕШНО" if r["ok"] else "ОШИБКА"

            f.write(f"Размер {r['size']}x{r['size']}\n")
            f.write(f"Статус: {status}\n")

            f.write(f"Файлы:\n")
            f.write(f"  A: {r['files'][0]}\n")
            f.write(f"  B: {r['files'][1]}\n")
            f.write(f"  C: {r['files'][2]}\n")


            f.write("\n")

    print("\nОтчет сохранен в verification_reports.txt")


def main():

    print("=" * 60)
    print("ПРОВЕРКА КОРРЕКТНОСТИ УМНОЖЕНИЯ МАТРИЦ")
    print("=" * 60)

    sizes = [200, 400, 800, 1200, 1600, 2000]

    results = []

    for n in sizes:

        res = check_result(n)

        if res is not None:
            results.append(res)

    if len(results) == 0:
        print("\nФайлы для проверки не найдены.")
        return

    print_summary(results)

    success = sum(1 for r in results if r["ok"])

    print("\nУспешных проверок:", success)
    print("Всего проверок:", len(results))

    if success == len(results):
        print("Все результаты корректны")
    else:
        print("Есть ошибки в вычислениях")

    save_report(results)


if __name__ == "__main__":
    main()