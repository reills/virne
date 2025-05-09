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
  id 1768
  arrival_time 39445.61949318147
  lifetime 555.1857863134096
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 12
    gpu 21
    rom 0
  ]
  node [
    id 1
    label "1"
    cpu 46
    gpu 11
    rom 24
  ]
  node [
    id 2
    label "2"
    cpu 49
    gpu 1
    rom 32
  ]
  node [
    id 3
    label "3"
    cpu 17
    gpu 37
    rom 39
  ]
  node [
    id 4
    label "4"
    cpu 23
    gpu 49
    rom 49
  ]
  node [
    id 5
    label "5"
    cpu 30
    gpu 11
    rom 27
  ]
  node [
    id 6
    label "6"
    cpu 23
    gpu 40
    rom 48
  ]
  node [
    id 7
    label "7"
    cpu 11
    gpu 36
    rom 36
  ]
  node [
    id 8
    label "8"
    cpu 8
    gpu 22
    rom 39
  ]
  node [
    id 9
    label "9"
    cpu 11
    gpu 35
    rom 39
  ]
  node [
    id 10
    label "10"
    cpu 15
    gpu 37
    rom 41
  ]
  node [
    id 11
    label "11"
    cpu 13
    gpu 11
    rom 28
  ]
  node [
    id 12
    label "12"
    cpu 23
    gpu 46
    rom 4
  ]
  edge [
    source 0
    target 1
    bw 34
  ]
  edge [
    source 1
    target 2
    bw 28
  ]
  edge [
    source 2
    target 3
    bw 18
  ]
  edge [
    source 3
    target 4
    bw 21
  ]
  edge [
    source 4
    target 5
    bw 7
  ]
  edge [
    source 5
    target 6
    bw 37
  ]
  edge [
    source 6
    target 7
    bw 17
  ]
  edge [
    source 7
    target 8
    bw 29
  ]
  edge [
    source 8
    target 9
    bw 38
  ]
  edge [
    source 9
    target 10
    bw 44
  ]
  edge [
    source 10
    target 11
    bw 6
  ]
  edge [
    source 11
    target 12
    bw 49
  ]
]
