graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 239
  arrival_time 4425.324030074613
  lifetime 190.2541580665723
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 11
    gpu 20
    rom 9
  ]
  node [
    id 1
    label "1"
    cpu 21
    gpu 14
    rom 28
  ]
  node [
    id 2
    label "2"
    cpu 14
    gpu 0
    rom 46
  ]
  node [
    id 3
    label "3"
    cpu 16
    gpu 48
    rom 28
  ]
  node [
    id 4
    label "4"
    cpu 3
    gpu 33
    rom 39
  ]
  node [
    id 5
    label "5"
    cpu 1
    gpu 25
    rom 7
  ]
  node [
    id 6
    label "6"
    cpu 47
    gpu 28
    rom 45
  ]
  node [
    id 7
    label "7"
    cpu 14
    gpu 36
    rom 23
  ]
  node [
    id 8
    label "8"
    cpu 3
    gpu 45
    rom 46
  ]
  node [
    id 9
    label "9"
    cpu 38
    gpu 20
    rom 17
  ]
  node [
    id 10
    label "10"
    cpu 39
    gpu 24
    rom 19
  ]
  node [
    id 11
    label "11"
    cpu 32
    gpu 8
    rom 14
  ]
  node [
    id 12
    label "12"
    cpu 21
    gpu 20
    rom 50
  ]
  edge [
    source 0
    target 1
    bw 40
  ]
  edge [
    source 1
    target 2
    bw 17
  ]
  edge [
    source 2
    target 3
    bw 26
  ]
  edge [
    source 3
    target 4
    bw 48
  ]
  edge [
    source 4
    target 5
    bw 2
  ]
  edge [
    source 5
    target 6
    bw 26
  ]
  edge [
    source 6
    target 7
    bw 29
  ]
  edge [
    source 7
    target 8
    bw 44
  ]
  edge [
    source 8
    target 9
    bw 13
  ]
  edge [
    source 9
    target 10
    bw 13
  ]
  edge [
    source 10
    target 11
    bw 47
  ]
  edge [
    source 11
    target 12
    bw 49
  ]
]
