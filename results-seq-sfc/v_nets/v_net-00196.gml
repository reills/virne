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
  id 196
  arrival_time 3537.2277255803424
  lifetime 1598.1933559359297
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 30
    gpu 12
    rom 14
  ]
  node [
    id 1
    label "1"
    cpu 17
    gpu 41
    rom 16
  ]
  node [
    id 2
    label "2"
    cpu 11
    gpu 23
    rom 48
  ]
  node [
    id 3
    label "3"
    cpu 49
    gpu 2
    rom 32
  ]
  node [
    id 4
    label "4"
    cpu 43
    gpu 4
    rom 12
  ]
  node [
    id 5
    label "5"
    cpu 16
    gpu 39
    rom 31
  ]
  node [
    id 6
    label "6"
    cpu 24
    gpu 45
    rom 24
  ]
  node [
    id 7
    label "7"
    cpu 19
    gpu 27
    rom 11
  ]
  node [
    id 8
    label "8"
    cpu 40
    gpu 18
    rom 6
  ]
  node [
    id 9
    label "9"
    cpu 44
    gpu 23
    rom 34
  ]
  node [
    id 10
    label "10"
    cpu 2
    gpu 40
    rom 32
  ]
  edge [
    source 0
    target 1
    bw 19
  ]
  edge [
    source 1
    target 2
    bw 36
  ]
  edge [
    source 2
    target 3
    bw 40
  ]
  edge [
    source 3
    target 4
    bw 34
  ]
  edge [
    source 4
    target 5
    bw 30
  ]
  edge [
    source 5
    target 6
    bw 49
  ]
  edge [
    source 6
    target 7
    bw 0
  ]
  edge [
    source 7
    target 8
    bw 34
  ]
  edge [
    source 8
    target 9
    bw 47
  ]
  edge [
    source 9
    target 10
    bw 18
  ]
]
