"""
===========================================
ТЕСТ 1: ПЕРЕВІРКА ЗАКОНУ ЗБЕРЕЖЕННЯ ЕНЕРГІЇ
===========================================

Цей скрипт перевіряє, чи зберігається енергія в системі під час симуляції.

ЩО РОБИТЬ:
- Запускає симуляцію Сонце-Земля на 365 днів (1 рік)
- Обчислює енергію кожні 1000 кроків
- Створює графік енергії
- Обчислює похибку

РЕЗУЛЬТАТИ ДЛЯ ЗВІТУ:
- Графік energy_conservation_test.png
- Текстовий звіт test_results.txt
===========================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Імпортуємо класи з основного коду
try:
    from main import CelestialBody, PhysicsEngine
except ImportError:
    print("Помилка: Не знайдено main.py")
    print("Скопіюйте класи CelestialBody та PhysicsEngine нижче")

#============================
# КЛАС ДЛЯ ТЕСТУВАННЯ ЕНЕРГІЇ
#============================
class EnergyTester:
    def __init__(self, physics_engine):
        self.physics_engine = physics_engine

        # Записи для графіка
        self.time_points = []
        self.kinetic_energy = []
        self.potential_energy = []
        self.total_energy = []

    def calculate_kinetic_energy(self):
        """
        Обчислює кінетичну енергію системи.
        Формула: E_k = (1/2) × m × v²
        """
        ke = 0.0
        for body in self.physics_engine.bodies:
            v = np.linalg.norm(body.velocity)  # модуль швидкості
            ke += 0.5 * body.mass * (v ** 2)
        return ke

    def calculate_potential_energy(self):
        """
        Обчислює потенційну енергію системи.
        Формула: E_p = -G × m1 × m2 / r
        (сума для всіх пар тіл)
        """
        pe = 0.0
        for i, body1 in enumerate(self.physics_engine.bodies):
            for body2 in self.physics_engine.bodies[i + 1:]:
                r = np.linalg.norm(body1.position - body2.position)
                if r > 0:
                    pe -= self.physics_engine.G * body1.mass * body2.mass / r
        return pe

    def calculate_total_energy(self):
        """Обчислює повну енергію: E = E_k + E_p"""
        ke = self.calculate_kinetic_energy()
        pe = self.calculate_potential_energy()
        return ke + pe

    def record_energy(self):
        """Записує поточні значення енергії"""
        ke = self.calculate_kinetic_energy()
        pe = self.calculate_potential_energy()
        te = ke + pe

        self.time_points.append(self.physics_engine.time)
        self.kinetic_energy.append(ke)
        self.potential_energy.append(pe)
        self.total_energy.append(te)

    def run_test(self, num_steps=36500, dt=1000.0, record_interval=100):
        """
        Запускає тест на певну кількість кроків.

        Параметри:
        ----------
        num_steps : int
            Кількість кроків симуляції (36500 = ~1 рік)
        dt : float
            Крок часу в секундах (1000 с)
        record_interval : int
            Записувати енергію кожні N кроків
        """
        print("ЗАПУСК ТЕСТУ ЗБЕРЕЖЕННЯ ЕНЕРГІЇ")
        print("=" * 60)
        print(f"Кількість кроків: {num_steps}")
        print(f"Крок часу: {dt} секунд")
        print(f"Тривалість: {num_steps * dt / 86400:.1f} днів")
        print("=" * 60)

        # Початкова енергія
        initial_energy = self.calculate_total_energy()
        print(f"Початкова повна енергія: {initial_energy:.6e} Дж")

        # Запис початкового стану
        self.record_energy()

        # Симуляція
        for step in range(1, num_steps + 1):
            self.physics_engine.update(dt)

            # Записуємо енергію
            if step % record_interval == 0:
                self.record_energy()

            # Прогрес
            if step % 5000 == 0:
                progress = (step / num_steps) * 100
                print(f"Прогрес: {progress:.1f}% (день {step * dt / 86400:.1f})")

        # Фінальна енергія
        final_energy = self.total_energy[-1]
        print("\n" + "=" * 60)
        print(f"Фінальна повна енергія: {final_energy:.6e} Дж")

        # Обчислюємо похибку
        energy_change = abs(final_energy - initial_energy)
        relative_error = (energy_change / abs(initial_energy)) * 100

        print(f"Зміна енергії: {energy_change:.6e} Дж")
        print(f"Відносна похибка: {relative_error:.4f}%")
        print("=" * 60)

        # Висновок
        if relative_error < 1.0:
            print("ТЕСТ ПРОЙДЕНО: Енергія зберігається (похибка < 1%)")
        elif relative_error < 5.0:
            print("ТЕСТ ПРОЙДЕНО З ЗАСТЕРЕЖЕННЯМ: Похибка 1-5%")
        else:
            print("ТЕСТ НЕ ПРОЙДЕНО: Похибка > 5%")

        return {
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'relative_error': relative_error,
            'passed': relative_error < 5.0
        }

    def plot_results(self, filename='energy_conservation_test.png'):
        """Створює графік енергії"""
        if len(self.time_points) < 2:
            print(" Недостатньо даних для графіку")
            return

        print(f"\n Створення графіку...")

        # Перетворюємо час у дні
        time_days = [t / 86400 for t in self.time_points]

        # Створюємо графік
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Верхній графік: Всі енергії
        ax1.plot(time_days, self.kinetic_energy,
                 label='Кінетична енергія', color='blue', linewidth=2)
        ax1.plot(time_days, self.potential_energy,
                 label='Потенційна енергія', color='red', linewidth=2)
        ax1.plot(time_days, self.total_energy,
                 label='Повна енергія', color='green', linewidth=2.5)

        ax1.set_xlabel('Час (дні)', fontsize=12)
        ax1.set_ylabel('Енергія (Дж)', fontsize=12)
        ax1.set_title('Тест збереження енергії: Сонце-Земля', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Нижній графік: Відхилення повної енергії
        initial_energy = self.total_energy[0]
        energy_deviation = [(e - initial_energy) / abs(initial_energy) * 100
                            for e in self.total_energy]

        ax2.plot(time_days, energy_deviation,
                 color='purple', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Час (дні)', fontsize=12)
        ax2.set_ylabel('Відхилення енергії (%)', fontsize=12)
        ax2.set_title('Відносна зміна повної енергії', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Графік збережено: {filename}")
        plt.close()

    def save_report(self, test_results, filename='test_results.txt'):
        """Зберігає текстовий звіт"""
        print(f"\n Збереження звіту...")

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("     ЗВІТ ТЕСТУВАННЯ: ЗАКОН ЗБЕРЕЖЕННЯ ЕНЕРГІЇ\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Дата тестування: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            f.write(f"Система: Сонце-Земля\n")
            f.write(f"Тривалість: {self.time_points[-1] / 86400:.1f} днів\n")
            f.write(f"Кількість вимірювань: {len(self.time_points)}\n\n")

            f.write("-" * 70 + "\n")
            f.write("РЕЗУЛЬТАТИ:\n")
            f.write("-" * 70 + "\n\n")

            f.write(f"Початкова повна енергія:  {test_results['initial_energy']:.6e} Дж\n")
            f.write(f"Фінальна повна енергія:   {test_results['final_energy']:.6e} Дж\n")
            f.write(
                f"Зміна енергії:            {abs(test_results['final_energy'] - test_results['initial_energy']):.6e} Дж\n")
            f.write(f"Відносна похибка:         {test_results['relative_error']:.4f}%\n\n")

            f.write("-" * 70 + "\n")
            f.write("ВИСНОВОК:\n")
            f.write("-" * 70 + "\n\n")

            if test_results['passed']:
                f.write("ТЕСТ ПРОЙДЕНО\n\n")
                f.write("Закон збереження енергії виконується з прийнятною точністю.\n")
                f.write("Відхилення менше 5%, що є нормальним для методу Ейлера\n")
                f.write("з обраним кроком часу.\n\n")
            else:
                f.write("ТЕСТ НЕ ПРОЙДЕНО\n\n")
                f.write("Похибка перевищує 5%. Рекомендації:\n")
                f.write("- Зменшити крок часу (dt)\n")
                f.write("- Використати метод Рунге-Кутта замість Ейлера\n\n")

            f.write("-" * 70 + "\n")
            f.write("ДЕТАЛЬНІ ДАНІ:\n")
            f.write("-" * 70 + "\n\n")

            f.write("Час (дні) | Кінетична (Дж) | Потенційна (Дж) | Повна (Дж) | Відхилення (%)\n")
            f.write("-" * 90 + "\n")

            initial = self.total_energy[0]
            for i in range(0, len(self.time_points), len(self.time_points) // 20):
                day = self.time_points[i] / 86400
                ke = self.kinetic_energy[i]
                pe = self.potential_energy[i]
                te = self.total_energy[i]
                dev = (te - initial) / abs(initial) * 100

                f.write(f"{day:8.1f} | {ke:14.6e} | {pe:15.6e} | {te:10.6e} | {dev:+8.4f}\n")

        print(f"Звіт збережено: {filename}")


#===========================
# ГОЛОВНА ФУНКЦІЯ ТЕСТУВАННЯ - запускає тест збереження енергії
#===========================
def main():
    print("\n" + "=" * 70)
    print("  ТЕСТ 1: ПЕРЕВІРКА ЗАКОНУ ЗБЕРЕЖЕННЯ ЕНЕРГІЇ")
    print("=" * 70 + "\n")

    # Створюємо фізичний рушій
    physics = PhysicsEngine()

    # Створюємо систему Сонце-Земля
    print("Створення системи Сонце-Земля...")

    sun = CelestialBody(
        name="Сонце",
        mass=1.989e30,
        position=[0, 0],
        velocity=[0, 0],
        color=(255, 220, 0)
    )
    physics.add_body(sun)

    earth = CelestialBody(
        name="Земля",
        mass=5.972e24,
        position=[1.496e11, 0],
        velocity=[0, 29780],
        color=(50, 150, 255)
    )
    physics.add_body(earth)

    print("Система створена\n")

    # Створюємо тестер
    tester = EnergyTester(physics)

    # Запускаємо тест
    # 36500 кроків × 1000 секунд = ~365 днів (1 рік)
    test_results = tester.run_test(
        num_steps=36500,  # Кількість кроків
        dt=1000.0,  # Крок часу (секунди)
        record_interval=100  # Записувати кожні 100 кроків
    )

    # Створюємо графік
    tester.plot_results('energy_conservation_test.png')

    # Зберігаємо звіт
    tester.save_report(test_results, 'test_results.txt')

    print("\n" + "=" * 70)
    print("ТЕСТУВАННЯ ЗАВЕРШЕНО")
    print("=" * 70)
    print("\n Створені файли:")
    print("   1. energy_conservation_test.png - графік енергії")
    print("   2. test_results.txt - текстовий звіт")


if __name__ == "__main__":
    main()