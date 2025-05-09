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
  id 363
  arrival_time 6874.450976508411
  lifetime 86.89837337801666
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 49
    gpu 19
    rom 33
  ]
  node [
    id 1
    label "1"
    cpu 49
    gpu 32
    rom 23
  ]
  node [
    id 2
    label "2"
    cpu 50
    gpu 7
    rom 35
  ]
  node [
    id 3
    label "3"
    cpu 28
    gpu 2
    rom 28
  ]
  node [
    id 4
    label "4"
    cpu 9
    gpu 25
    rom 19
  ]
  node [
    id 5
    label "5"
    cpu 37
    gpu 30
    rom 44
  ]
  node [
    id 6
    label "6"
    cpu 8
    gpu 3
    rom 49
  ]
  node [
    id 7
    label "7"
    cpu 11
    gpu 48
    rom 49
  ]
  node [
    id 8
    label "8"
    cpu 12
    gpu 25
    rom 39
  ]
  node [
    id 9
    label "9"
    cpu 44
    gpu 3
    rom 36
  ]
  node [
    id 10
    label "10"
    cpu 9
    gpu 17
    rom 33
  ]
  edge [
    source 0
    target 1
    bw 22
  ]
  edge [
    source 1
    target 2
    bw 21
  ]
  edge [
    source 2
    target 3
    bw 35
  ]
  edge [
    source 3
    target 4
    bw 27
  ]
  edge [
    source 4
    target 5
    bw 33
  ]
  edge [
    source 5
    target 6
    bw 1
  ]
  edge [
    source 6
    target 7
    bw 45
  ]
  edge [
    source 7
    target 8
    bw 35
  ]
  edge [
    source 8
    target 9
    bw 10
  ]
  edge [
    source 9
    target 10
    bw 25
  ]
]
