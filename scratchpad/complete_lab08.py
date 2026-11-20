import json
import shutil
import sys
import uuid
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parent.parent
path = root / 'lab_08.ipynb'
backup = root / 'scratchpad' / 'lab_08_before_completion.ipynb'
if not backup.exists():
    shutil.copyfile(path, backup)
nb = nbformat.read(path, as_version=4)

def code(text):
    return nbformat.v4.new_code_cell(text.strip())

def md(text):
    return nbformat.v4.new_markdown_cell(text.strip())

cells = nb.cells[:15]
cells[13].source = "print('Количество изображений с кошками:', np.sum(y_hog == 1))\nprint('Количество изображений с собаками:', np.sum(y_hog == 0))"
cells.insert(14, md('## Таблица HOG-признаков\n\nКаждая строка соответствует изображению. Имя файла и метка сохранены вместе с признаками; при обучении используются только числовые HOG-признаки.'))
cells.insert(15, code('''
df_hog = pd.DataFrame(
    X_hog,
    columns=[f'hog_{i}' for i in range(X_hog.shape[1])]
)
df_hog.insert(0, 'filename', filenames)
df_hog['label'] = y_hog
df_hog.to_csv('lab_08_hog.csv', index=False, encoding='utf-8-sig')

print('Размер таблицы:', df_hog.shape)
display(df_hog.head())
'''))
cells.insert(16, md('## Разделение на train и test\n\nОставляем 20% изображений для теста. Стратификация сохраняет доли классов. Все модели используют одно разбиение.'))

cells += [
md('''## Сравнение трёх методов классификации

Сравним SVM с RBF-ядром, KNN и Random Forest. Для SVM и KNN масштабирование включено в Pipeline: StandardScaler обучается только на train. Параметры заданы до оценки на test, подбор по тестовой выборке не выполняется.

В таблице приведены accuracy и macro-усреднённые precision, recall, F1. Для визуального разбора выберем модель с наибольшей accuracy, при равенстве — с наибольшим F1 macro. Это описательное сравнение на одном разбиении; для независимой оценки выбранной модели понадобились бы новые тестовые данные.'''),
code('''
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class_names = ['Собака', 'Кошка']  # метки 0 и 1

models = {
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1, gamma='scale'))
    ]),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=1
    )
}

predictions = {}
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    rows.append({
        'model': name,
        'accuracy': accuracy_score(y_test, pred),
        'precision_macro': precision_score(y_test, pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, pred, average='macro', zero_division=0)
    })

results = pd.DataFrame(rows).sort_values(
    ['accuracy', 'f1_macro'], ascending=False, kind='stable'
).reset_index(drop=True)
results.to_csv('lab_08_model_comparison.csv', index=False, encoding='utf-8-sig')
display(results.round(3))

best_name = results.loc[0, 'model']
best_model = models[best_name]
y_pred = predictions[best_name]
print('Модель для разбора:', best_name)
'''),
md('## Отчёты и матрицы ошибок\n\nВезде используется порядок: собака — 0, кошка — 1.'),
code('''
for name, pred in predictions.items():
    print(name)
    print(classification_report(
        y_test, pred, labels=[0, 1], target_names=class_names, zero_division=0
    ))

fig, axes = plt.subplots(1, len(models), figsize=(16, 4))
for ax, (name, pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(name)
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
plt.tight_layout()
plt.show()
'''),
md('## Визуальный разбор предсказаний\n\nПоказываем все ошибки выбранной модели и по пять верных предсказаний каждого класса. На изображениях указаны истинная и предсказанная метки, а также имя файла.'),
code('''
errors = np.flatnonzero(y_test != y_pred)
correct = np.flatnonzero(y_test == y_pred)
print('Модель:', best_name)
print('Количество ошибок:', len(errors), 'из', len(y_test))

test_predictions = pd.DataFrame({
    'filename': filenames[test_idx],
    'true_label': y_test,
    'predicted_label': y_pred,
    'correct': y_test == y_pred
})
display(test_predictions)

def show_predictions(positions, title):
    if len(positions) == 0:
        print(title + ': примеров нет')
        return
    ncols = min(4, len(positions))
    nrows = (len(positions) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for ax, pos in zip(axes.flat, positions):
        original_idx = test_idx[pos]
        ax.imshow(X_resized[original_idx])
        ax.set_title(
            f'{filenames[original_idx]}\\n'
            f'Истина: {class_names[y_test[pos]]}\\n'
            f'Прогноз: {class_names[y_pred[pos]]}', fontsize=10
        )
        ax.axis('off')
    for ax in axes.flat[len(positions):]:
        ax.axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

show_predictions(errors, f'Ошибки: {best_name}')
correct_examples = np.concatenate([
    correct[y_test[correct] == label][:5] for label in [0, 1]
])
show_predictions(correct_examples, f'Верные предсказания: {best_name}')
'''),
]
nb.cells = cells
for c in nb.cells:
    if c.cell_type == 'code':
        c.outputs = []
        c.execution_count = None
nbformat.write(nb, path)

# Use the same interpreter as the lab's virtual environment.
km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(nb, km=km, timeout=180, resources={'metadata': {'path': str(root)}})
try:
    client.execute()
finally:
    nbformat.write(nb, path)

# Export the actual notebook plots for visual review.
import base64
for ci, cell in enumerate(nb.cells):
    for oi, output in enumerate(cell.get('outputs', [])):
        data = output.get('data', {})
        if 'image/png' in data and ci >= len(nb.cells) - 3:
            (root / 'scratchpad' / f'lab08_review_{ci}_{oi}.png').write_bytes(base64.b64decode(data['image/png']))

print('Notebook executed and saved.')
print((root / 'lab_08_model_comparison.csv').read_text(encoding='utf-8-sig'))
