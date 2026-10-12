; The ten words of the game's own code that change. Each becomes a call to the new code, which
; does what the word did for every other leader. See punch.s and jump.s for the far side of each.

; ==== the walk state, where it moves a leader ====
.org 0x60c4fe14                         ; the call to the steering, once a frame
    bl    j_walk

; ==== entering the punch state ====
.org 0x60c55978                         ; "the clip to play is the punch"
    bl    j_enter_a
.org 0x60c559d4                         ; the same in the state's second way in
    bl    j_enter_b

; ==== each frame of the punch state ====
.org 0x60c55a48                         ; "is this the frame it lands?"
    bl    j_flags
.org 0x60c55aa4                         ; the radius of the sphere searched for an enemy, 10.0
    bl    j_reach
.org 0x60c55b14                         ; the furthest away a hit may land, 20.0
    bl    j_furthest
.org 0x60c55c30                         ; the strength of the hit, 3.0
    bl    j_strength
.org 0x60c55c70                         ; the call that hands the enemy the hit
    bl    j_send
.org 0x60c55f28                         ; "has the clip ended?"
    bl    j_update
.org 0x60c55fd0                         ; the same question on the path taken after a kill
    bl    j_update
