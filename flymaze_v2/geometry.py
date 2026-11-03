"""Geometry in cell units. Walls are closed unit squares, the body is a disk."""

import math


def point_segment_sq(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px-ax)*dx+(py-ay)*dy)/length_sq))
    return (px-ax-t*dx)**2 + (py-ay-t*dy)**2


def ray_box(x, y, dx, dy, left, top, right, bottom):
    """First nonnegative ray entry into a rectangle; infinity on a miss."""
    near, far = 0.0, math.inf
    for origin, direction, low, high in ((x, dx, left, right), (y, dy, top, bottom)):
        if abs(direction) < 1e-12:
            if origin < low or origin > high:
                return math.inf
        else:
            a, b = (low-origin)/direction, (high-origin)/direction
            near, far = max(near, min(a, b)), min(far, max(a, b))
            if far < near:
                return math.inf
    return near


def segment_box_sq(ax, ay, bx, by, left, top, right, bottom):
    if ray_box(ax, ay, bx-ax, by-ay, left, top, right, bottom) <= 1.0:
        return 0.0
    def point_box_sq(x, y):
        return max(left-x, 0, x-right)**2 + max(top-y, 0, y-bottom)**2
    return min(point_box_sq(ax, ay), point_box_sq(bx, by),
               *(point_segment_sq(x, y, ax, ay, bx, by)
                 for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))))


def segment_circle_entry(ax, ay, bx, by, cx, cy, radius):
    """First segment fraction in a circle, or None."""
    ox, oy, dx, dy = ax-cx, ay-cy, bx-ax, by-ay
    c = ox*ox + oy*oy - radius*radius
    if c <= 0:
        return 0.0
    a = dx*dx + dy*dy
    if a == 0:
        return None
    b = 2*(ox*dx + oy*dy)
    discriminant = b*b-4*a*c
    if discriminant < 0:
        return None
    t = (-b-math.sqrt(discriminant))/(2*a)
    return t if 0 <= t <= 1 else None
