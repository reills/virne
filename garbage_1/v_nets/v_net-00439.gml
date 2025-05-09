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
  id 439
  arrival_time 8452.862444805543
  lifetime 1267.6255758271052
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 46
    gpu 45
    rom 19
  ]
  node [
    id 1
    label "1"
    cpu 12
    gpu 20
    rom 45
  ]
  node [
    id 2
    label "2"
    cpu 20
    gpu 23
    rom 23
  ]
  node [
    id 3
    label "3"
    cpu 26
    gpu 4
    rom 24
  ]
  node [
    id 4
    label "4"
    cpu 12
    gpu 22
    rom 35
  ]
  node [
    id 5
    label "5"
    cpu 34
    gpu 28
    rom 49
  ]
  node [
    id 6
    label "6"
    cpu 27
    gpu 35
    rom 33
  ]
  node [
    id 7
    label "7"
    cpu 9
    gpu 12
    rom 16
  ]
  node [
    id 8
    label "8"
    cpu 23
    gpu 21
    rom 0
  ]
  node [
    id 9
    label "9"
    cpu 4
    gpu 26
    rom 7
  ]
  node [
    id 10
    label "10"
    cpu 4
    gpu 32
    rom 38
  ]
  node [
    id 11
    label "11"
    cpu 37
    gpu 15
    rom 21
  ]
  edge [
    source 0
    target 1
    bw 17
  ]
  edge [
    source 1
    target 2
    bw 26
  ]
  edge [
    source 2
    target 3
    bw 0
  ]
  edge [
    source 3
    target 4
    bw 3
  ]
  edge [
    source 4
    target 5
    bw 21
  ]
  edge [
    source 5
    target 6
    bw 41
  ]
  edge [
    source 6
    target 7
    bw 4
  ]
  edge [
    source 7
    target 8
    bw 33
  ]
  edge [
    source 8
    target 9
    bw 15
  ]
  edge [
    source 9
    target 10
    bw 31
  ]
  edge [
    source 10
    target 11
    bw 29
  ]
]
