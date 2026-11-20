import hashlib
import io
import json
import random
import uuid
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'lab_08_data' / 'cats_vs_dogs'
ARCHIVE = ROOT / 'scratchpad' / 'cats_and_dogs_filtered.zip'
SOURCE = 'https://download.mlcc.google.com/mledu-datasets/cats_and_dogs_filtered.zip'
TUTORIAL = 'https://www.tensorflow.org/tutorials/images/transfer_learning'

DEST.mkdir(parents=True, exist_ok=True)
records = []
seen = set()
rng = random.Random(42)
with zipfile.ZipFile(ARCHIVE) as archive:
    for category, label in [('cats', 1), ('dogs', 0)]:
        folder = DEST / category
        folder.mkdir(exist_ok=True)
        candidates = sorted(name for name in archive.namelist()
                            if name.startswith(f'cats_and_dogs_filtered/train/{category}/')
                            and name.lower().endswith('.jpg'))
        rng.shuffle(candidates)
        selected = []
        for name in candidates:
            data = archive.read(name)
            with Image.open(io.BytesIO(data)) as image:
                image = image.convert('RGB')
                image.load()
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                width, height = image.size
            if digest in seen:
                continue
            seen.add(digest)
            target = folder / Path(name).name
            target.write_bytes(data)
            record = dict(file=f'{category}/{target.name}', label=label,
                          archive_member=name, sha256=hashlib.sha256(data).hexdigest(),
                          width=width, height=height)
            records.append(record)
            selected.append(target)
            if len(selected) == 100:
                break
        assert len(selected) == 100
        # Contact sheet for checking class labels and mixed scenes.
        sheet = Image.new('RGB', (1200, 1400), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(selected):
            with Image.open(path) as image:
                thumb = ImageOps.contain(image.convert('RGB'), (116, 116))
                x, y = (i % 10) * 120, (i // 10) * 140
                sheet.paste(thumb, (x, y))
                draw.text((x, y + 118), path.name, fill='black')
        sheet.save(ROOT / 'scratchpad' / f'{category}_contact.jpg')

metadata = dict(source_url=SOURCE, source_page=TUTORIAL, seed=42,
                description='100 cats (label 1), 100 dogs without cats (label 0).',
                selection='Random sample from the train folders, with decoded exact duplicates excluded.',
                images=records)
(DEST / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

p = ROOT / 'lab_08.ipynb'
n = json.loads(p.read_text(encoding='utf-8'))

def set_code(index, source):
    cell = n['cells'][index]
    cell['source'] = source.splitlines(keepends=True)
    cell['outputs'] = []
    cell['execution_count'] = None

set_code(3, '''from pathlib import Path

# Объект распознавания — кошка. Отрицательный класс — собаки.
data_dir = Path('lab_08_data') / 'cats_vs_dogs'
cats_dir = data_dir / 'cats'
dogs_dir = data_dir / 'dogs'

for folder in [cats_dir, dogs_dir]:
    if not folder.is_dir():
        raise FileNotFoundError(f'Не найдена папка: {folder.resolve()}')
''')
set_code(4, '''# Читаем изображения, сохраняем имена файлов и метки классов.
X = []
y = []
filenames = []

for folder, label in [(cats_dir, 1), (dogs_dir, 0)]:
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    for path in files:
        image = imread(path)
        X.append(image)
        y.append(label)
        filenames.append(path.relative_to(data_dir).as_posix())

y = np.array(y)
filenames = np.array(filenames)

print('Всего изображений:', len(X))
print('Кошек (класс 1):', np.sum(y == 1))
print('Собак (класс 0):', np.sum(y == 0))

assert len(X) == 200
assert np.sum(y == 1) == 100 and np.sum(y == 0) == 100
''')
n['cells'].insert(3, dict(cell_type='markdown', id=uuid.uuid4().hex[:8], metadata={}, source=[
    '## Датасет: есть кошка / нет кошки\n',
    '\n',
    'Собрана выборка из 200 фотографий: 100 кошек (метка 1) и 100 собак (метка 0). '
    'Собаки выбраны как тематически близкий отрицательный класс.\n',
    '\n',
    f'Источник: [Cats and Dogs, учебный пример TensorFlow]({TUTORIAL}). '
    'Из папок train исходного архива случайно выбрано по 100 изображений; seed=42. '
    'Это подвыборка готового набора, а не собственные фотографии. '
    'Список файлов, исходные пути и контрольные суммы сохранены в '
    '`lab_08_data/cats_vs_dogs/manifest.json`.\n'
]))
p.write_text(json.dumps(n, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Prepared dataset and updated lab_08.ipynb: 100 cats + 100 dogs.')
