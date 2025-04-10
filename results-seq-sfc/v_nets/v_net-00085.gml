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
  id 85
  arrival_time 1667.4702273863154
  lifetime 2697.1676724475533
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 50
    gpu 24
    rom 8
  ]
  node [
    id 1
    label "1"
    cpu 50
    gpu 0
    rom 15
  ]
  node [
    id 2
    label "2"
    cpu 44
    gpu 16
    rom 0
  ]
  node [
    id 3
    label "3"
    cpu 11
    gpu 36
    rom 10
  ]
  node [
    id 4
    label "4"
    cpu 22
    gpu 32
    rom 46
  ]
  node [
    id 5
    label "5"
    cpu 38
    gpu 9
    rom 29
  ]
  node [
    id 6
    label "6"
    cpu 38
    gpu 34
    rom 9
  ]
  node [
    id 7
    label "7"
    cpu 4
    gpu 20
    rom 40
  ]
  node [
    id 8
    label "8"
    cpu 41
    gpu 32
    rom 29
  ]
  node [
    id 9
    label "9"
    cpu 9
    gpu 11
    rom 23
  ]
  node [
    id 10
    label "10"
    cpu 37
    gpu 7
    rom 49
  ]
  node [
    id 11
    label "11"
    cpu 21
    gpu 25
    rom 21
  ]
  node [
    id 12
    label "12"
    cpu 47
    gpu 22
    rom 5
  ]
  edge [
    source 0
    target 1
    bw 32
  ]
  edge [
    source 1
    target 2
    bw 45
  ]
  edge [
    source 2
    target 3
    bw 8
  ]
  edge [
    source 3
    target 4
    bw 21
  ]
  edge [
    source 4
    target 5
    bw 27
  ]
  edge [
    source 5
    target 6
    bw 6
  ]
  edge [
    source 6
    target 7
    bw 45
  ]
  edge [
    source 7
    target 8
    bw 42
  ]
  edge [
    source 8
    target 9
    bw 38
  ]
  edge [
    source 9
    target 10
    bw 41
  ]
  edge [
    source 10
    target 11
    bw 28
  ]
  edge [
    source 11
    target 12
    bw 42
  ]
]
