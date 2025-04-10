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
  id 1730
  arrival_time 38661.91243461322
  lifetime 501.9838169095235
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 29
    gpu 28
    rom 33
  ]
  node [
    id 1
    label "1"
    cpu 15
    gpu 37
    rom 7
  ]
  node [
    id 2
    label "2"
    cpu 36
    gpu 31
    rom 1
  ]
  node [
    id 3
    label "3"
    cpu 35
    gpu 16
    rom 26
  ]
  node [
    id 4
    label "4"
    cpu 27
    gpu 38
    rom 39
  ]
  node [
    id 5
    label "5"
    cpu 4
    gpu 31
    rom 50
  ]
  node [
    id 6
    label "6"
    cpu 47
    gpu 12
    rom 7
  ]
  node [
    id 7
    label "7"
    cpu 43
    gpu 19
    rom 15
  ]
  node [
    id 8
    label "8"
    cpu 22
    gpu 21
    rom 41
  ]
  node [
    id 9
    label "9"
    cpu 25
    gpu 47
    rom 23
  ]
  node [
    id 10
    label "10"
    cpu 17
    gpu 10
    rom 32
  ]
  node [
    id 11
    label "11"
    cpu 50
    gpu 50
    rom 39
  ]
  node [
    id 12
    label "12"
    cpu 21
    gpu 29
    rom 37
  ]
  node [
    id 13
    label "13"
    cpu 29
    gpu 32
    rom 33
  ]
  edge [
    source 0
    target 1
    bw 47
  ]
  edge [
    source 1
    target 2
    bw 13
  ]
  edge [
    source 2
    target 3
    bw 7
  ]
  edge [
    source 3
    target 4
    bw 37
  ]
  edge [
    source 4
    target 5
    bw 42
  ]
  edge [
    source 5
    target 6
    bw 24
  ]
  edge [
    source 6
    target 7
    bw 18
  ]
  edge [
    source 7
    target 8
    bw 30
  ]
  edge [
    source 8
    target 9
    bw 49
  ]
  edge [
    source 9
    target 10
    bw 18
  ]
  edge [
    source 10
    target 11
    bw 32
  ]
  edge [
    source 11
    target 12
    bw 7
  ]
  edge [
    source 12
    target 13
    bw 47
  ]
]
