; bringing_problem.pddl
(define (problem bring_muffin)
  (:domain bringing)

  (:objects
    tiago - robot
    muffin - object
    kitchen living_room user_location - location
  )

  (:init
    (robot_at tiago living_room)
    (at muffin kitchen)
    (hand_empty tiago)
  )

  (:goal
    (at muffin user_location)
  )
)