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
  id 1630
  arrival_time 36558.847623116126
  lifetime 239.1124685810398
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 44
    gpu 21
    rom 9
  ]
  node [
    id 1
    label "1"
    cpu 1
    gpu 25
    rom 25
  ]
  node [
    id 2
    label "2"
    cpu 42
    gpu 6
    rom 22
  ]
  node [
    id 3
    label "3"
    cpu 15
    gpu 15
    rom 26
  ]
  node [
    id 4
    label "4"
    cpu 28
    gpu 24
    rom 20
  ]
  node [
    id 5
    label "5"
    cpu 10
    gpu 14
    rom 44
  ]
  node [
    id 6
    label "6"
    cpu 22
    gpu 32
    rom 11
  ]
  node [
    id 7
    label "7"
    cpu 7
    gpu 30
    rom 11
  ]
  node [
    id 8
    label "8"
    cpu 45
    gpu 50
    rom 18
  ]
  node [
    id 9
    label "9"
    cpu 2
    gpu 13
    rom 32
  ]
  node [
    id 10
    label "10"
    cpu 45
    gpu 33
    rom 19
  ]
  edge [
    source 0
    target 1
    bw 28
  ]
  edge [
    source 1
    target 2
    bw 4
  ]
  edge [
    source 2
    target 3
    bw 31
  ]
  edge [
    source 3
    target 4
    bw 47
  ]
  edge [
    source 4
    target 5
    bw 4
  ]
  edge [
    source 5
    target 6
    bw 18
  ]
  edge [
    source 6
    target 7
    bw 31
  ]
  edge [
    source 7
    target 8
    bw 48
  ]
  edge [
    source 8
    target 9
    bw 10
  ]
  edge [
    source 9
    target 10
    bw 17
  ]
]
