# Навчальний проєкт 2. Розподілене моделювання клітинного автомата

Виконала Черниченко Надія, ТТП-42. Дисципліна: «Методи паралельних обчислень».

## Склад проєкту

- `src/fire.c` - V0, смугова V1/V3, небезпечний обмін V2, 2D, overlap, MPI+OpenMP.
- `src/deadlock.c` - дослідження тупика та два безпечні способи обміну.
- `src/channel.c` - вимірювання параметрів каналу.
- `scripts/experiments.py` - перевірка коректності й усі серії вимірювань.
- `scripts/plots.py` - графіки за `results.csv`.
- `scripts/animation.py`, `scripts/visualize.py` - анімація та зображення сітки.
- `scripts/percolation_local.py` - незалежна перевірка перколяції.
- `measurements/` - журнал запусків, конфігурація і початкові дані перколяції.
- `figures/` - графіки та анімація; `Звіт.pdf` - готовий звіт.
- `results.csv` - результати всіх дослідів, включно з прогрівами та очікуваними тайм-аутами.

## Середовище та збирання

Перевірено на Ubuntu 26.04 LTS у WSL2, Open MPI 5.0.10, GCC 15.2.0. Комп’ютер: AMD Ryzen 5 7430U, 6 фізичних ядер і 12 логічних процесорів. Деталі містяться в `measurements/environment.json` і `host_*.json`.

Команди виконуються в Ubuntu з каталогу `fire_project`:

```bash
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev
make
ompi_info --version
mpicc --version
```

Параметри збирання: `-O3 -std=c11 -Wall -Wextra -fopenmp`, компонування `-lm -fopenmp`. У цій версії пакета `mpirun --version` може повідомляти про відсутній файл довідки; `ompi_info --version` працює.

## Запуск і перевірка сітки

```bash
mpirun --use-hwthread-cpus --map-by hwthread --bind-to hwthread -np 1 ./fire v0 64 0.60 42 measurements/v0.bin
mpirun --use-hwthread-cpus --map-by hwthread --bind-to hwthread -np 4 ./fire stripe 64 0.60 42 ordered measurements/p4.bin
cmp measurements/v0.bin measurements/p4.bin
```

У смуговій версії доступні методи `ordered`, `nonblocking`, `overlap`, `hybrid`, `bad`. Для двовимірної декомпозиції:

```bash
mpirun --use-hwthread-cpus --map-by hwthread --bind-to hwthread -np 4 ./fire 2d 512 0.60 42
```

Порівняння MPI та MPI+OpenMP на тих самих 6 фізичних ядрах:

```bash
mpirun --map-by slot:PE=1 --bind-to core -np 6 ./fire stripe 512 0.60 42 nonblocking
OMP_NUM_THREADS=2 OMP_PROC_BIND=true OMP_PLACES=cores mpirun --map-by slot:PE=2 --bind-to core -np 3 ./fire stripe 512 0.60 42 hybrid
```

## Тупик і канал

Небезпечний обмін запускається з обмеженням часу. На дослідженій системі 4016 байтів проходять, 4017 байтів спричиняють тупик. Для іншої системи поріг визначається заново.

```bash
timeout -k 2s 5s mpirun --use-hwthread-cpus --map-by hwthread --bind-to hwthread -np 2 ./deadlock 4017 100 bad
mpirun --use-hwthread-cpus --map-by hwthread --bind-to hwthread -np 2 ./deadlock 8192 1000 ordered
mpirun --use-hwthread-cpus --map-by hwthread --bind-to hwthread -np 2 ./deadlock 8192 1000 nonblocking
mpirun --use-hwthread-cpus --map-by hwthread --bind-to hwthread -np 2 ./channel 1000
```

Код `124` від timeout у небезпечному досліді означає очікуване перевищення ліміту.

## Відтворення вимірювань

```bash
python3 scripts/experiments.py --n 1472 --weak-n 384 --extra-n 512 --repeats 3
```

Спочатку порівнюються вся сітка побайтово й усі підсумкові показники для різних N, d, seed і P. Після помилки або розбіжності сценарій зупиняється. Далі виконуються пошук порога тупика, сильне й слабке масштабування, додаткові варіанти, порівняння на однакових фізичних ядрах і вимірювання каналу.

Перед кожною основною серією виконується прогрів і три вимірювані повторення. Розмір сильного досліду перевіряється та за потреби змінюється до 20-60 с для V0. Для слабкого досліду `N=round(384*sqrt(P))`; площа на процес стала, число кроків може змінюватися.

Нові дані спочатку записуються в `results_new.csv`; `results.csv` замінюється після завершення всіх серій. Перколяційні записи додаються з `measurements/percolation_original.csv`. Це 1050 незалежних перевірок чотирисусідної зв’язності, еквівалентної проходженню вогню, а не вимірювання часу MPI. Сирі команди й вивід збережено в `measurements/raw_runs.jsonl`.

## Графіки та візуалізація

Python-залежності наведено в `requirements.txt`. У середовищі з установленими пакетами:

```bash
python scripts/plots.py
python scripts/animation.py --n 120 --density 0.60 --seed 42 --output figures/поширення.gif
python scripts/visualize.py measurements/v0.bin --n 64 --output figures/final_grid.png
```

`percolation_local.py` записує тільки `measurements/percolation_original.csv` і не перезаписує загальні результати.

## Подання

Готові матеріали: `Звіт.pdf`, `results.csv`, README, Makefile, вихідні коди, сценарії, графіки та журнал вимірювань. Архів для подання: `Черниченко_Проєкт2.zip`.

Усі п’ять додаткових напрямів представлені у звіті. Сумарний бонус обмежено 5 балами, загальна оцінка - 30; остаточне оцінювання включає захист. Методичні обмеження вимірювань зазначено в самому звіті.
