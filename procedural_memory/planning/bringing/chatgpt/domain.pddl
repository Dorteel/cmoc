; bringing_domain.pddl
(define (domain bringing)
  (:requirements :strips :typing)

  (:types robot object location)

  (:predicates
    (robot_at ?r - robot ?l - location)
    (at ?o - object ?l - location)
    (holding ?r - robot ?o - object)
    (hand_empty ?r - robot)
  )

  (:action move
    :parameters (?r - robot ?from - location ?to - location)
    :precondition (robot_at ?r ?from)
    :effect (and
      (not (robot_at ?r ?from))
      (robot_at ?r ?to)
    )
  )

  (:action pick
    :parameters (?r - robot ?o - object ?l - location)
    :precondition (and
      (robot_at ?r ?l)
      (at ?o ?l)
      (hand_empty ?r)
    )
    :effect (and
      (holding ?r ?o)
      (not (at ?o ?l))
      (not (hand_empty ?r))
    )
  )

  (:action place
    :parameters (?r - robot ?o - object ?l - location)
    :precondition (and
      (robot_at ?r ?l)
      (holding ?r ?o)
    )
    :effect (and
      (at ?o ?l)
      (hand_empty ?r)
      (not (holding ?r ?o))
    )
  )
)