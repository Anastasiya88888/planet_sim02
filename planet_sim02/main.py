"""
СИМУЛЯТОР ПЛАНЕТАРНОЇ СИСТЕМИ
ВСТАНОВЛЕННЯ:
    pip install numpy pygame

ЗАПУСК:
    python main.py

МОЖЛИВОСТІ:
    * Моделювання руху планет за законами Ньютона
    * 3 готові системи (Сонячна, Подвійна зірка, Три тіла)
    * Додавання власних планет
    * Зміна швидкості симуляції (1x - 10x)
    * Експорт звіту у .txt файл
    * Візуалізація орбіт у реальному часі

ПОМІТКА:
    Цифри краще вводити на англійській клавіатурі (бо українська 'е' не сприймається програмою)
"""

import numpy as np
import pygame
from datetime import datetime


#====================================
# КЛАС: CelestialBody (Небесне тіло) - Представляє небесне тіло (зірку, планету). Зберігає масу, позицію, швидкість та історію руху.
#====================================
class CelestialBody:
    def __init__(self, name, mass, position, velocity, color=(255, 255, 255)):
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.array([0.0, 0.0])
        self.color = color
        self.trajectory = []  # Історія для малювання орбіти

    def apply_force(self, force):
        """Застосовує силу: a = F / m"""
        self.acceleration = force / self.mass

    def update(self, dt):
        """Оновлює стан за методом Ейлера"""
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

        self.trajectory.append(self.position.copy())
        if len(self.trajectory) > 500:
            self.trajectory.pop(0)


#=====================================
# КЛАС: PhysicsEngine (Фізичний рушій) - Обчислює гравітаційні взаємодії за законом Ньютона - [F = G × (m1 × m2) / r²].
#=====================================
class PhysicsEngine:
    G = 6.67430e-11  # Гравітаційна стала

    def __init__(self):
        self.bodies = []
        self.time = 0.0

    def add_body(self, body):
        """Додає тіло до системи"""
        self.bodies.append(body)

    def remove_body(self, body):
        """Видаляє тіло"""
        if body in self.bodies:
            self.bodies.remove(body)

    def clear_all(self):
        """Очищує систему"""
        self.bodies.clear()
        self.time = 0.0

    def calculate_gravitational_force(self, body1, body2):
        """Обчислює силу між двома тілами"""
        direction = body2.position - body1.position
        distance = np.linalg.norm(direction)

        if distance < 1e3:
            return np.array([0.0, 0.0])

        direction_unit = direction / distance
        force_magnitude = self.G * body1.mass * body2.mass / (distance ** 2)

        return force_magnitude * direction_unit

    def update(self, dt):
        """Оновлює всі тіла на один крок часу"""
        # Крок 1: Обчислюємо сили
        for body in self.bodies:
            total_force = np.array([0.0, 0.0])
            for other_body in self.bodies:
                if body != other_body:
                    force = self.calculate_gravitational_force(body, other_body)
                    total_force += force
            body.apply_force(total_force)

        # Крок 2: Оновлюємо позиції
        for body in self.bodies:
            body.update(dt)

        self.time += dt


#======================
# КЛАС: Camera (Камера) - Перетворює світові координати в екранні.
#======================

class Camera:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.zoom = 1.0
        self.offset = np.array([0.0, 0.0])

    def world_to_screen(self, position):
        """Метри → Пікселі"""
        screen_x = (position[0] + self.offset[0]) * self.zoom + self.screen_width / 2
        screen_y = (position[1] + self.offset[1]) * self.zoom + self.screen_height / 2
        return int(screen_x), int(screen_y)

    def adjust_zoom_for_system(self, bodies):
        #Автоматичне масштабування
        if not bodies:
            return

        max_distance = 0
        for body in bodies:
            distance = np.linalg.norm(body.position)
            if distance > max_distance:
                max_distance = distance

        if max_distance > 0:
            screen_size = min(self.screen_width, self.screen_height)
            self.zoom = (screen_size * 0.4) / max_distance


#==============================
# КЛАС: Renderer (Візуалізатор) - Малює симуляцію на екрані
#==============================

class Renderer:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    def __init__(self, screen_width=1200, screen_height=700):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Симулятор планетарної системи")

        self.camera = Camera(screen_width, screen_height)
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)

    def draw_body(self, body):
        """Малює планету"""
        screen_pos = self.camera.world_to_screen(body.position)

        if (0 <= screen_pos[0] <= self.screen_width and
                0 <= screen_pos[1] <= self.screen_height):
            radius = max(3, min(30, int(5 + np.log10(body.mass) / 2)))
            pygame.draw.circle(self.screen, body.color, screen_pos, radius)

            name_text = self.font_small.render(body.name, True, self.WHITE)
            self.screen.blit(name_text, (screen_pos[0] + radius + 5, screen_pos[1] - 10))

    def draw_trajectory(self, body):
        """Малює орбіту"""
        if len(body.trajectory) < 2:
            return

        screen_points = []
        for pos in body.trajectory:
            screen_pos = self.camera.world_to_screen(pos)
            if (0 <= screen_pos[0] <= self.screen_width and
                    0 <= screen_pos[1] <= self.screen_height):
                screen_points.append(screen_pos)

        if len(screen_points) >= 2:
            trajectory_color = tuple(max(0, c - 50) for c in body.color)
            pygame.draw.lines(self.screen, trajectory_color, False, screen_points, 1)

    def draw_info(self, physics_engine, sim_speed, is_paused):
        """Виводить інформацію"""
        info_lines = [
            f"Час: {physics_engine.time / 86400:.1f} діб",
            f"Об'єктів: {len(physics_engine.bodies)}",
            f"Швидкість: {sim_speed}x",
            f"{'ПАУЗА' if is_paused else 'АКТИВНО'}"
        ]

        # Напівпрозорий фон
        s = pygame.Surface((300, 110))
        s.set_alpha(200)
        s.fill((20, 20, 20))
        self.screen.blit(s, (5, 5))
        pygame.draw.rect(self.screen, self.WHITE, pygame.Rect(5, 5, 300, 110), 1)

        y_offset = 15
        for line in info_lines:
            text = self.font_medium.render(line, True, self.WHITE)
            self.screen.blit(text, (15, y_offset))
            y_offset += 25

    def render(self, physics_engine, sim_speed, is_paused):
        """Головна функція малювання"""
        self.screen.fill(self.BLACK)

        self.camera.adjust_zoom_for_system(physics_engine.bodies)

        for body in physics_engine.bodies:
            self.draw_trajectory(body)

        for body in physics_engine.bodies:
            self.draw_body(body)

        self.draw_info(physics_engine, sim_speed, is_paused)

        pygame.display.flip()


#========================
# КЛАС: UIButton (Кнопка)
#========================

class UIButton:
    def __init__(self, x, y, width, height, text, color=(100, 100, 200)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = tuple(min(255, c + 40) for c in color)
        self.font = pygame.font.Font(None, 22)
        self.is_hovered = False

    def draw(self, screen):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)

        text_surface = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


#=================
# КЛАС: InputField - Поле для введення тексту
#=================
class InputField:
    def __init__(self, x, y, width, height, label="", default_value=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.value = default_value
        self.active = False
        self.font = pygame.font.Font(None, 18)

    def draw(self, screen):
        color = (200, 200, 255) if self.active else (150, 150, 150)
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2 if self.active else 1)

        if self.label:
            label_surf = self.font.render(self.label, True, (255, 255, 255))
            screen.blit(label_surf, (self.rect.x, self.rect.y - 18))

        text_surf = self.font.render(self.value, True, (0, 0, 0))
        screen.blit(text_surf, (self.rect.x + 5, self.rect.y + 6))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.value = self.value[:-1]
            elif event.key == pygame.K_RETURN:
                self.active = False
            elif len(self.value) < 20:
                self.value += event.unicode


#================================================
# КЛАС: SimulationController (Головний контролер) - Керує всім застосунком
#================================================
class SimulationController:
    def __init__(self):
        # Ядро
        self.physics_engine = PhysicsEngine()
        self.renderer = Renderer(1200, 700)

        # Стан
        self.is_running = True
        self.is_paused = True
        self.simulation_speed = 1
        self.dt = 1000.0  # крок часу в секундах

        # Кнопки
        self.buttons = {
            'start': UIButton(10, 650, 80, 35, 'Старт', (50, 150, 50)),
            'pause': UIButton(100, 650, 80, 35, 'Пауза', (150, 150, 50)),
            'stop': UIButton(190, 650, 80, 35, 'Стоп', (150, 50, 50)),
            'add': UIButton(400, 650, 80, 35, 'Додати', (50, 100, 200)),
            'clear': UIButton(490, 650, 80, 35, 'Очистити', (200, 50, 50)),
            'export': UIButton(580, 650, 80, 35, 'Експорт', (100, 100, 100)),
            'speed_1x': UIButton(850, 650, 50, 35, '1x'),
            'speed_2x': UIButton(910, 650, 50, 35, '2x'),
            'speed_5x': UIButton(970, 650, 50, 35, '5x'),
            'speed_10x': UIButton(1030, 650, 60, 35, '10x'),
            'preset_solar': UIButton(10, 130, 110, 30, 'Сонячна', (100, 150, 200)),
            'preset_binary': UIButton(10, 170, 110, 30, 'Подвійна', (200, 100, 150)),
            'preset_three': UIButton(10, 210, 110, 30, 'Три тіла', (150, 200, 100)),
        }

        # Поля введення
        self.input_fields = {
            'name': InputField(300, 600, 140, 25, 'Назва:', 'Планета'),
            'mass': InputField(450, 600, 140, 25, 'Маса (кг):', '5.972e24'),
            'distance': InputField(600, 600, 140, 25, 'Відстань (м):', '1.5e11'),
            'speed': InputField(750, 600, 140, 25, 'Швидкість (м/с):', '30000'),
            'angle': InputField(900, 600, 100, 25, 'Кут (°):', '0'),
        }

        self.clock = pygame.time.Clock()
        self.fps = 60

        # Завантажуємо початкову систему
        self.load_example_system()

        print("Застосунок запущено")
        print("=" * 60)
        print("Натисніть кнопки пресетів зліва для готових систем!")
        print("=" * 60)

    def load_example_system(self):
        """Проста система Сонце-Земля"""
        sun = CelestialBody("Сонце", 1.989e30, [0, 0], [0, 0], (255, 220, 0))
        self.physics_engine.add_body(sun)

        earth = CelestialBody("Земля", 5.972e24, [1.496e11, 0], [0, 29780], (50, 150, 255))
        self.physics_engine.add_body(earth)

        print("Завантажено: Сонце-Земля")

    #--------------
    # ПРЕСЕТИ СИСТЕМ
    #---------------

    def load_solar_system(self):
        """ПРЕСЕТ 1: Сонячна система (внутрішні планети)"""
        self.physics_engine.clear_all()
        self.is_paused = True

        # Сонце
        sun = CelestialBody("Сонце", 1.989e30, [0, 0], [0, 0], (255, 220, 0))
        self.physics_engine.add_body(sun)

        # Меркурій
        mercury = CelestialBody("Меркурій", 3.3e23, [5.79e10, 0], [0, 47900], (169, 169, 169))
        self.physics_engine.add_body(mercury)

        # Венера
        venus = CelestialBody("Венера", 4.87e24, [1.08e11, 0], [0, 35000], (255, 198, 73))
        self.physics_engine.add_body(venus)

        # Земля
        earth = CelestialBody("Земля", 5.972e24, [1.496e11, 0], [0, 29780], (50, 150, 255))
        self.physics_engine.add_body(earth)

        # Марс
        mars = CelestialBody("Марс", 6.39e23, [2.28e11, 0], [0, 24000], (255, 100, 50))
        self.physics_engine.add_body(mars)

        print("ПРЕСЕТ: Сонячна система (4 планети)")

    def load_binary_stars(self):
        """ПРЕСЕТ 2: Подвійна зоряна система"""
        self.physics_engine.clear_all()
        self.is_paused = True

        m = 1e30  # маса зірки
        d = 1e11  # відстань між зірками
        v = np.sqrt(self.physics_engine.G * m / d)  # орбітальна швидкість

        star_a = CelestialBody("Зірка A", m, [d / 2, 0], [0, v], (255, 100, 100))
        self.physics_engine.add_body(star_a)

        star_b = CelestialBody("Зірка B", m, [-d / 2, 0], [0, -v], (100, 100, 255))
        self.physics_engine.add_body(star_b)

        print("ПРЕСЕТ: Подвійна зоряна система")

    def load_three_body(self):
        """ПРЕСЕТ 3: Три тіла (трикутна конфігурація)"""
        self.physics_engine.clear_all()
        self.is_paused = True

        m = 1e30
        r = 1.5e11
        v = np.sqrt(self.physics_engine.G * m * 1.577 / r)

        # Три тіла на вершинах рівностороннього трикутника
        body1 = CelestialBody(
            "Тіло 1", m,
            [r * np.cos(np.pi / 2), r * np.sin(np.pi / 2)],
            [-v * np.sin(np.pi / 2), v * np.cos(np.pi / 2)],
            (255, 100, 100)
        )
        self.physics_engine.add_body(body1)

        body2 = CelestialBody(
            "Тіло 2", m,
            [r * np.cos(7 * np.pi / 6), r * np.sin(7 * np.pi / 6)],
            [-v * np.sin(7 * np.pi / 6), v * np.cos(7 * np.pi / 6)],
            (100, 255, 100)
        )
        self.physics_engine.add_body(body2)

        body3 = CelestialBody(
            "Тіло 3", m,
            [r * np.cos(-np.pi / 6), r * np.sin(-np.pi / 6)],
            [-v * np.sin(-np.pi / 6), v * np.cos(-np.pi / 6)],
            (100, 100, 255)
        )
        self.physics_engine.add_body(body3)

        print("ПРЕСЕТ: Система трьох тіл")

    #-----------------
    # ДОДАВАННЯ ПЛАНЕТ - додає планету з полів введення.
    #-----------------

    def add_planet_from_input(self):
        try:
            name = self.input_fields['name'].value
            mass = float(self.input_fields['mass'].value)
            distance = float(self.input_fields['distance'].value)
            speed = float(self.input_fields['speed'].value)
            angle_deg = float(self.input_fields['angle'].value)

            angle_rad = np.deg2rad(angle_deg)

            position = [
                distance * np.cos(angle_rad),
                distance * np.sin(angle_rad)
            ]

            velocity = [
                -speed * np.sin(angle_rad),
                speed * np.cos(angle_rad)
            ]

            color = (
                np.random.randint(100, 255),
                np.random.randint(100, 255),
                np.random.randint(100, 255)
            )

            planet = CelestialBody(name, mass, position, velocity, color)
            self.physics_engine.add_body(planet)

            print(f"Додано: {name}")

        except ValueError as e:
            print(f"Помилка введення: {e}")

    #--------------
    # ЕКСПОРТ ЗВІТУ - експортує звіт у .txt файл.
    #--------------

    def export_report(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("  ЗВІТ СИМУЛЯЦІЇ ПЛАНЕТАРНОЇ СИСТЕМИ\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            f.write(f"Час симуляції: {self.physics_engine.time:.2f} с ")
            f.write(f"({self.physics_engine.time / 86400:.2f} діб)\n")
            f.write(f"Кількість об'єктів: {len(self.physics_engine.bodies)}\n\n")
            f.write("-" * 60 + "\n\n")

            for i, body in enumerate(self.physics_engine.bodies, 1):
                f.write(f"{i}. {body.name}\n")
                f.write(f"   Маса: {body.mass:.3e} кг\n")
                f.write(f"   Позиція: ({body.position[0]:.3e}, {body.position[1]:.3e}) м\n")
                f.write(f"   Швидкість: {np.linalg.norm(body.velocity):.2f} м/с\n")
                f.write(f"   Відстань від центру: {np.linalg.norm(body.position):.3e} м\n\n")

        print(f"Звіт збережено: {filename}")

    #--------------
    # ОБРОБКА ПОДІЙ
    #--------------

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False

            # Поля введення
            for field in self.input_fields.values():
                field.handle_event(event)

            # Кнопки керування
            if self.buttons['start'].handle_event(event):
                self.is_paused = False
                print("▶ Старт")

            if self.buttons['pause'].handle_event(event):
                self.is_paused = True
                print("⏸ Пауза")

            if self.buttons['stop'].handle_event(event):
                self.is_paused = True
                self.physics_engine.clear_all()
                self.load_example_system()
                print("⏹ Стоп")

            if self.buttons['add'].handle_event(event):
                self.add_planet_from_input()

            if self.buttons['clear'].handle_event(event):
                self.physics_engine.clear_all()
                print("🗑 Очищено")

            if self.buttons['export'].handle_event(event):
                self.export_report()

            # Швидкість
            if self.buttons['speed_1x'].handle_event(event):
                self.simulation_speed = 1
            if self.buttons['speed_2x'].handle_event(event):
                self.simulation_speed = 2
            if self.buttons['speed_5x'].handle_event(event):
                self.simulation_speed = 5
            if self.buttons['speed_10x'].handle_event(event):
                self.simulation_speed = 10

            # ПРЕСЕТИ
            if self.buttons['preset_solar'].handle_event(event):
                self.load_solar_system()

            if self.buttons['preset_binary'].handle_event(event):
                self.load_binary_stars()

            if self.buttons['preset_three'].handle_event(event):
                self.load_three_body()

    def update(self):
        """Оновлює симуляцію"""
        if not self.is_paused:
            for _ in range(self.simulation_speed):
                self.physics_engine.update(self.dt)

    def render(self):
        """Малює все"""
        self.renderer.render(self.physics_engine, self.simulation_speed, self.is_paused)

        for button in self.buttons.values():
            button.draw(self.renderer.screen)

        for field in self.input_fields.values():
            field.draw(self.renderer.screen)

        pygame.display.flip()

    def run(self):
        """Головний цикл"""
        while self.is_running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(self.fps)

        pygame.quit()
        print("\n" + "=" * 60)
        print("Дякуємо за використання!")
        print("=" * 60)

#================
# ЗАПУСК ПРОГРАМИ
#================

if __name__ == "__main__":
    try:
        app = SimulationController()
        app.run()
    except KeyboardInterrupt:
        print("\n Перервано користувачем")
    except Exception as e:
        print(f"\n Помилка: {e}")
        import traceback
        traceback.print_exc()