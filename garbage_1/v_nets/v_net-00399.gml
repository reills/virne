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
  id 399
  arrival_time 7843.558700782731
  lifetime 30.043253071215844
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 4
    gpu 3
    rom 50
  ]
  node [
    id 1
    label "1"
    cpu 29
    gpu 47
    rom 42
  ]
  node [
    id 2
    label "2"
    cpu 26
    gpu 7
    rom 11
  ]
  node [
    id 3
    label "3"
    cpu 6
    gpu 21
    rom 14
  ]
  node [
    id 4
    label "4"
    cpu 9
    gpu 50
    rom 19
  ]
  node [
    id 5
    label "5"
    cpu 36
    gpu 12
    rom 15
  ]
  node [
    id 6
    label "6"
    cpu 26
    gpu 29
    rom 29
  ]
  node [
    id 7
    label "7"
    cpu 30
    gpu 24
    rom 47
  ]
  node [
    id 8
    label "8"
    cpu 34
    gpu 36
    rom 34
  ]
  node [
    id 9
    label "9"
    cpu 45
    gpu 11
    rom 0
  ]
  node [
    id 10
    label "10"
    cpu 50
    gpu 43
    rom 2
  ]
  node [
    id 11
    label "11"
    cpu 46
    gpu 1
    rom 13
  ]
  node [
    id 12
    label "12"
    cpu 15
    gpu 17
    rom 33
  ]
  edge [
    source 0
    target 1
    bw 20
  ]
  edge [
    source 1
    target 2
    bw 12
  ]
  edge [
    source 2
    target 3
    bw 5
  ]
  edge [
    source 3
    target 4
    bw 13
  ]
  edge [
    source 4
    target 5
    bw 18
  ]
  edge [
    source 5
    target 6
    bw 17
  ]
  edge [
    source 6
    target 7
    bw 16
  ]
  edge [
    source 7
    target 8
    bw 21
  ]
  edge [
    source 8
    target 9
    bw 35
  ]
  edge [
    source 9
    target 10
    bw 10
  ]
  edge [
    source 10
    target 11
    bw 29
  ]
  edge [
    source 11
    target 12
    bw 21
  ]
]
