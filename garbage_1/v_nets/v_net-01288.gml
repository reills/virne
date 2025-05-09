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
  id 1288
  arrival_time 26835.64533152104
  lifetime 112.88785212732351
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 40
    gpu 13
    rom 47
  ]
  node [
    id 1
    label "1"
    cpu 27
    gpu 42
    rom 3
  ]
  node [
    id 2
    label "2"
    cpu 40
    gpu 36
    rom 3
  ]
  node [
    id 3
    label "3"
    cpu 10
    gpu 37
    rom 31
  ]
  node [
    id 4
    label "4"
    cpu 25
    gpu 24
    rom 33
  ]
  node [
    id 5
    label "5"
    cpu 18
    gpu 10
    rom 44
  ]
  node [
    id 6
    label "6"
    cpu 33
    gpu 42
    rom 38
  ]
  node [
    id 7
    label "7"
    cpu 3
    gpu 26
    rom 18
  ]
  node [
    id 8
    label "8"
    cpu 40
    gpu 25
    rom 25
  ]
  node [
    id 9
    label "9"
    cpu 14
    gpu 6
    rom 11
  ]
  edge [
    source 0
    target 1
    bw 50
  ]
  edge [
    source 1
    target 2
    bw 0
  ]
  edge [
    source 2
    target 3
    bw 32
  ]
  edge [
    source 3
    target 4
    bw 47
  ]
  edge [
    source 4
    target 5
    bw 0
  ]
  edge [
    source 5
    target 6
    bw 50
  ]
  edge [
    source 6
    target 7
    bw 44
  ]
  edge [
    source 7
    target 8
    bw 5
  ]
  edge [
    source 8
    target 9
    bw 15
  ]
]
