from typing import List, Optional, Tuple, Dict
import math


class Basket:
    def __init__(self, type_id: int, basket_weight: int, cost_fruits: Dict[str, float], price_baskets: Dict[int, float], apple: int = 0, peach: int = 0, nectarine: int = 0):
        self.type_id = type_id
        self.basket_weight = basket_weight
        self.cost_fruits = cost_fruits
        self.price_baskets = price_baskets
        self.apple = apple
        self.peach = peach
        self.nectarine = nectarine

    def total(self) -> int:
        return self.apple + self.peach + self.nectarine

    def remaining(self) -> int:
        return self.basket_weight - self.total()

    def min_apple(self) -> int:
        return {1: math.ceil(0.30 * self.basket_weight), 2: 0, 3: math.ceil(0.20 * self.basket_weight)}[self.type_id]

    def max_peach(self) -> int:
        return {1: math.floor(0.20 * self.basket_weight), 2: math.floor(0.40 * self.basket_weight), 3: self.basket_weight}[self.type_id]

    def min_nectarine(self) -> int:
        return {1: 0, 2: math.ceil(0.20 * self.basket_weight), 3: 0}[self.type_id]

    def max_nectarine(self) -> int:
        return {1: self.basket_weight, 2: self.basket_weight, 3: math.floor(0.30 * self.basket_weight)}[self.type_id]

    def is_complete(self) -> bool:
        return self.total() == self.basket_weight and self.is_feasible()

    def is_feasible(self) -> bool:
        if self.peach > self.max_peach():
            return False
        if self.nectarine > self.max_nectarine():
            return False
        if self.total() != self.basket_weight:
            return False
        if self.apple < self.min_apple():
            return False
        if self.nectarine < self.min_nectarine():
            return False
        return True

    def is_potentially_feasible(self) -> bool:
        if self.peach > self.max_peach():
            return False
        if self.nectarine > self.max_nectarine():
            return False
        remaining_capacity = self.remaining()
        if remaining_capacity < 0:
            return False
        if self.apple + remaining_capacity < self.min_apple():
            return False
        if self.nectarine + remaining_capacity < self.min_nectarine():
            return False
        if self.total() > self.basket_weight:
            return False
        return True

    def can_add(self, fruit: str) -> bool:
        if self.remaining() <= 0:
            return False
        test_basket = self.copy()
        if fruit == "apple":
            test_basket.apple += 1
        elif fruit == "peach":
            test_basket.peach += 1
        elif fruit == "nectarine":
            test_basket.nectarine += 1
        else:
            return False
        return test_basket.is_potentially_feasible()

    def add(self, fruit: str) -> None:
        assert self.can_add(fruit), f"Cannot add {fruit} to type {self.type_id} basket"
        if fruit == "apple":
            self.apple += 1
        elif fruit == "peach":
            self.peach += 1
        else:
            self.nectarine += 1

    def copy(self):
        return Basket(self.type_id, self.apple, self.peach, self.nectarine)

    def revenue(self) -> float:
        return self.price_baskets[self.type_id] * self.basket_weight if self.total() == self.basket_weight else self.price_baskets[self.type_id] * self.total()

    def cost(self) -> float:
        return self.apple * self.cost_fruits["apple"] + self.peach * self.cost_fruits["peach"] + self.nectarine * self.cost_fruits["nectarine"]

    def profit(self) -> float:
        return self.revenue() - self.cost()

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.type_id, self.apple, self.peach, self.nectarine)

    def __str__(self) -> str:
        return f"T{self.type_id}: A={self.apple}, P={self.peach}, N={self.nectarine} (r={self.remaining()})"
    

class Node:
    def __init__(self, basket: Basket, basket_weight: int, cost_fruits: Dict[str, float], price_baskets: Dict[int, float], supply: Dict[str, int], depth: int, action: Optional[str] = None):
        self.basket = basket
        self.basket_weight = basket_weight
        self.cost_fruits = cost_fruits
        self.price_baskets = price_baskets
        self.supply = supply
        self.depth = depth
        self.action = action
        self.children: List['Node'] = []
        self.profit_bound = 0.0

    def expand(self) -> List['Node']:
        if self.basket.remaining() == 0:
            return []
        children_nodes = []

        required_apples = max(0, self.basket.min_apple() - self.basket.apple)
        required_nectarines = max(0, self.basket.min_nectarine() - self.basket.nectarine)
        remaining_capacity = self.basket.remaining()
        forced_fruit = None

        if required_apples > 0 and required_apples == remaining_capacity:
            forced_fruit = "apple"
        if required_nectarines > 0 and required_nectarines == remaining_capacity:
            forced_fruit = "nectarine"

        fruits_order = ["apple", "peach", "nectarine"]
        if forced_fruit:
            fruits_order = [forced_fruit]

        for fruit in fruits_order:
            if self.supply.get(fruit, 0) <= 0:
                continue
            if not self.basket.can_add(fruit):
                continue

            new_basket = self.basket.copy()
            new_basket.add(fruit)
            new_supply = dict(self.supply)
            new_supply[fruit] -= 1

            child_node = Node(new_basket, new_supply, self.depth + 1, action=fruit)
            optimistic_cost_addition = new_basket.remaining() * self.cost_fruits["apple"]
            optimistic_profit = self.price_baskets[new_basket.type_id] * self.basket_weight - (new_basket.cost() + optimistic_cost_addition)
            child_node.profit_bound = optimistic_profit
            children_nodes.append(child_node)

        children_nodes.sort(key=lambda c: c.profit_bound, reverse=True)
        self.children = children_nodes
        return children_nodes
    

def build_best_basket_for_type(type_id: int, supply: Dict[str, int], cost_fruits: Dict[str, float], price_baskets: Dict[int, float], basket_weight: int, max_nodes: int = 2000) -> Optional[Tuple[Basket, Dict[str, int]]]:
    root = Node(
        basket=Basket(type_id, basket_weight=basket_weight, cost_fruits=cost_fruits, price_baskets=price_baskets),
        supply=dict(supply), depth=0,
        basket_weight=basket_weight, cost_fruits=cost_fruits, price_baskets=price_baskets
    )
    stack = [root]
    best: Optional[Node] = None
    visited = 0

    while stack and visited < max_nodes:
        node = stack.pop()
        visited += 1
        if node.basket.is_complete():
            if best is None or node.basket.profit() > best.basket.profit():
                best = node
            continue
        if best is not None and node.profit_bound <= best.basket.profit():
            continue
        children = node.expand()
        for child in reversed(children):
            stack.append(child)

    if best is None:
        return None
    return best.basket, best.supply


def heuristic_divide_and_conquer(initial_supply: Dict[str, int], cost_fruits: Dict[str, float], price_baskets: Dict[int, float], basket_weight: int) -> Tuple[List[Basket], Dict[str, int], float]:
    supply = dict(initial_supply)
    solution: List[Basket] = []

    while True:
        candidates: List[Tuple[Basket, Dict[str, int]]] = []
        for t in (1, 2, 3):
            res = build_best_basket_for_type(t, supply, cost_fruits, price_baskets, basket_weight)
            if res is not None:
                candidates.append(res)
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0].profit(), reverse=True)
        best_basket, new_supply = candidates[0]
        solution.append(best_basket)
        supply = new_supply

    total_profit = sum(b.profit() for b in solution)
    return solution, supply, total_profit
