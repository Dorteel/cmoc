(define (domain searching)
 (:requirements :strips :typing)
 (:types agent location)
 (:predicates (checked ?l - location))
 (:action look-at
  :parameters (?a - agent ?l - location)
  :precondition (and)
  :effect (checked ?l)))
