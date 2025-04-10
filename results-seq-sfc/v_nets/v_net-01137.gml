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
  id 1137
  arrival_time 23793.735556296837
  lifetime 484.0762504975488
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 21
    gpu 39
    rom 21
  ]
  node [
    id 1
    label "1"
    cpu 0
    gpu 40
    rom 46
  ]
  node [
    id 2
    label "2"
    cpu 20
    gpu 2
    rom 37
  ]
  node [
    id 3
    label "3"
    cpu 0
    gpu 37
    rom 50
  ]
  node [
    id 4
    label "4"
    cpu 26
    gpu 34
    rom 27
  ]
  node [
    id 5
    label "5"
    cpu 23
    gpu 24
    rom 48
  ]
  node [
    id 6
    label "6"
    cpu 37
    gpu 6
    rom 30
  ]
  node [
    id 7
    label "7"
    cpu 17
    gpu 23
    rom 42
  ]
  node [
    id 8
    label "8"
    cpu 15
    gpu 27
    rom 31
  ]
  node [
    id 9
    label "9"
    cpu 8
    gpu 44
    rom 49
  ]
  node [
    id 10
    label "10"
    cpu 49
    gpu 22
    rom 45
  ]
  edge [
    source 0
    target 1
    bw 31
  ]
  edge [
    source 1
    target 2
    bw 31
  ]
  edge [
    source 2
    target 3
    bw 40
  ]
  edge [
    source 3
    target 4
    bw 21
  ]
  edge [
    source 4
    target 5
    bw 29
  ]
  edge [
    source 5
    target 6
    bw 17
  ]
  edge [
    source 6
    target 7
    bw 14
  ]
  edge [
    source 7
    target 8
    bw 40
  ]
  edge [
    source 8
    target 9
    bw 36
  ]
  edge [
    source 9
    target 10
    bw 41
  ]
]
