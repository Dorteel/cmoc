; bringing_problem.pddl
(define (problem bring_standard)
  (:domain bringing)

  (:objects
    agent - robot
    theme - object
    source - location
    destination - location
    robot_start_location - location
  )

  (:init
    (robot_at agent robot_start_location)
    (at theme source)
    (hand_empty agent)
  )

  (:goal
    (at theme destination)
  )
)