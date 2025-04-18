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
  id 303
  arrival_time 5793.256854847985
  lifetime 1569.1178692034323
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 5
    gpu 48
    rom 49
  ]
  node [
    id 1
    label "1"
    cpu 36
    gpu 3
    rom 13
  ]
  node [
    id 2
    label "2"
    cpu 24
    gpu 36
    rom 43
  ]
  node [
    id 3
    label "3"
    cpu 9
    gpu 28
    rom 39
  ]
  node [
    id 4
    label "4"
    cpu 33
    gpu 17
    rom 32
  ]
  node [
    id 5
    label "5"
    cpu 46
    gpu 3
    rom 16
  ]
  node [
    id 6
    label "6"
    cpu 16
    gpu 34
    rom 3
  ]
  node [
    id 7
    label "7"
    cpu 29
    gpu 44
    rom 10
  ]
  node [
    id 8
    label "8"
    cpu 37
    gpu 16
    rom 6
  ]
  node [
    id 9
    label "9"
    cpu 2
    gpu 26
    rom 36
  ]
  edge [
    source 0
    target 1
    bw 20
  ]
  edge [
    source 1
    target 2
    bw 34
  ]
  edge [
    source 2
    target 3
    bw 6
  ]
  edge [
    source 3
    target 4
    bw 1
  ]
  edge [
    source 4
    target 5
    bw 43
  ]
  edge [
    source 5
    target 6
    bw 5
  ]
  edge [
    source 6
    target 7
    bw 28
  ]
  edge [
    source 7
    target 8
    bw 29
  ]
  edge [
    source 8
    target 9
    bw 30
  ]
]
